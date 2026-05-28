#!/usr/bin/env python3
"""
Skrypt do inżynierii sekwencji DNA wykorzystujący Atrybucję Gradientową (Top-K).
Zapewnia powtarzalność wyników dzięki globalnemu ustawieniu ziarna losowości (seed).
Dodano twardy limit mutacji dla zadania B.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn

SEQUENCE_LENGTH = 230


def set_seed(seed: int = 3) -> None:
    """
    Ustawia ziarno losowości dla wszystkich wykorzystywanych bibliotek.
    Gwarantuje, że przy tym samym modelu i danych wejściowych, optymalizator
    zawsze wybierze dokładnie te same mutacje w kolejnych krokach.
    Wymusza deterministyczne algorytmy w rdzeniu cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SEBlock(nn.Module):
    """
    Blok Squeeze-and-Excitation (Uwaga Kanałowa).
    Mechanizm oceniający, które z 96 wyekstrahowanych kanałów (cech) są w danym
    momencie najważniejsze.
    - Squeeze: Kompresuje informacje z całej sekwencji (AdaptiveAvgPool1d) do jednego wektora.
    - Excitation: Dwie warstwy konwolucyjne z nieliniowością ReLU i Sigmoid obliczają wagi (0 do 1) dla każdego kanału.
    Ostatecznie blok mnoży wejście przez te wagi, wyciszając kanały stanowiące szum.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skaluje oryginalną mapę cech przez wyuczone wagi od 0 do 1."""
        return x * self.se(x)


class ResidualDilatedBlock(nn.Module):
    """
    Rezydualny blok ekstrakcji cech z dylatacją.
    Dylatacja (dilation) to odległość między elementami przetwarzanymi przez jądro filtru.
    Pozwala to sieci "patrzeć" na coraz szersze fragmenty sekwencji DNA, wychwytując
    współdziałanie odległych motywów, nie zwiększając przy tym liczby wag modelu.
    Zawiera połączenie resztkowe (skip-connection), zapobiegające degradacji gradientów.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            SEBlock(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Dodaje wejście wejściowe do wyniku bloku (połączenie resztkowe F(x) + x)."""
        return self.activation(inputs + self.block(inputs))


class ResidualDilatedMultitaskCnn(nn.Module):
    """
    Odtworzona architektura głównej sieci CNN, identyczna z tą użytą podczas uczenia.
    - stem: Warstwa wejściowa lokalizująca podstawowe k-mery.
    - residual_blocks: Seria 5 bloków z wykładniczo rosnącą dylatacją.
    - pooling: Agreguje macierz przestrzenną za pomocą średniej (częstość występowania motywu) i maksimum (fakt wystąpienia motywu).
    - multitask heads: Rozgałęzienie na zadanie regresji i klasyfikacji. Do optymalizacji wykorzystujemy tylko regression_head.
    """

    def __init__(self, dropout: float = 0.20, channels: int = 96, dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
                 dense_units: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualDilatedBlock(channels, dilation, dropout) for dilation in dilations])
        self.shared_dense = nn.Sequential(
            nn.Linear(channels * 2, dense_units),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(dense_units, 1)
        self.classification_head = nn.Linear(dense_units, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Przetwarza macierz sekwencji i zwraca wynik rna_dna_ratio oraz logity is_active."""
        features = self.residual_blocks(self.stem(inputs))
        max_pooled = torch.amax(features, dim=2)
        avg_pooled = torch.mean(features, dim=2)
        shared = self.shared_dense(torch.cat((max_pooled, avg_pooled), dim=1))
        return self.regression_head(shared), self.classification_head(shared)


def one_hot_encode(sequences: list[str]) -> np.ndarray:
    """
    Tworzy macierze reprezentujące sekwencje ciągów znakowych, zrozumiałe dla sieci neuronowej.
    Dla każdej sekwencji tworzy tablicę 4x230. Pojedynczy nukleotyd jest oznaczany jako 1
    w wierszu odpowiadającym jego bazie (A=0, C=1, G=2, T=3), reszta to zera.
    """
    features = np.zeros((len(sequences), 4, SEQUENCE_LENGTH), dtype=np.float32)
    channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row_index, sequence in enumerate(sequences):
        for col_index, base in enumerate(sequence.upper()):
            if base in channel_by_base:
                features[row_index, channel_by_base[base], col_index] = 1.0
    return features


def select_device() -> torch.device:
    """Weryfikuje konfigurację sprzętową, wybierając GPU w przypadku posiadania sterowników CUDA."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta(file_path: str | Path) -> dict[str, str]:
    """
    Ekstraktor danych z formatu FASTA.
    Oddziela linie oznaczające nagłówki (zaczynające się od '>') od faktycznych sekwencji DNA,
    łącząc linie nukleotydów, które mogą być złamane (np. po 80 znakach).
    Zwraca słownik: {nazwa_sekwencji: ciąg_znaków}.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"BŁĄD: Nie znaleziono pliku {file_path}!")
        return {}

    sequences = {}
    with open(path, 'r', encoding='utf-8') as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.upper())
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    return sequences


def optimize_by_gradient(model: torch.nn.Module, start_seq: str, device: torch.device, max_mutations: int | None = None,
                         k_mutations: int = 5) -> tuple[str, list[float], np.ndarray]:
    """
    Główny algorytm sztucznej inteligencji służący do inżynierii wejścia (Atrybucja Gradientowa z Top-K).

    1. Koduje sekwencję (One-Hot) flagując ją jako parametr, względem którego będziemy liczyć błąd.
    2. Ewaluuje wynik przez sieć i wykonuje wsteczną propagację na tym wyniku.
    3. Macierz gradientów mówi wprost: "wstawienie 'T' na pozycję 12 zwiększy wynik predykcji".
    4. Analizuje każdą z 690 potencjalnych permutacji dla 230 bp.
    5. Formuje zbiór K najlepszych niezależnych modyfikacji, wprowadzając całą grupę na raz
       (rozwiązywanie problemu epistazy - zależności wielonukleotydowych).
    6. Zapętla operację aż do wyczerpania dostępnego budżetu `max_mutations` lub aż model oceni,
       że żadna pojedyncza zmiana nie zdoła już zwiększyć aktywności wzmacniacza (maksimum).
    """
    bases = ['A', 'C', 'G', 'T']
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    current_seq = list(start_seq.upper())
    mutations_done = 0
    history = []
    initial_gradients = None

    while True:
        if max_mutations is not None and mutations_done >= max_mutations:
            break

        features_np = one_hot_encode(["".join(current_seq)])
        features = torch.tensor(features_np, device=device, requires_grad=True)

        pred_ratio, _ = model(features)
        history.append(pred_ratio.item())

        model.zero_grad()
        pred_ratio.backward()

        gradients = features.grad.cpu().numpy()[0]
        if initial_gradients is None:
            initial_gradients = gradients.copy()

        possible_mutations = []

        for pos in range(len(current_seq)):
            current_base = current_seq[pos]
            current_idx = base_to_idx.get(current_base, -1)
            if current_idx == -1:
                continue

            for candidate_base in bases:
                if candidate_base == current_base:
                    continue

                candidate_idx = base_to_idx[candidate_base]
                improvement = gradients[candidate_idx, pos] - gradients[current_idx, pos]

                if improvement > 1e-6:
                    possible_mutations.append((improvement, pos, candidate_base))

        if not possible_mutations:
            break

        possible_mutations.sort(reverse=True, key=lambda x: x[0])

        applied_this_step = 0
        mutated_positions = set()

        for imp, pos, base in possible_mutations:
            if pos not in mutated_positions:
                current_seq[pos] = base
                mutated_positions.add(pos)
                mutations_done += 1
                applied_this_step += 1

                if applied_this_step >= k_mutations or (max_mutations is not None and mutations_done >= max_mutations):
                    break

        if applied_this_step == 0:
            break

    final_features_np = one_hot_encode(["".join(current_seq)])
    final_features = torch.tensor(final_features_np, device=device)
    with torch.no_grad():
        final_ratio, _ = model(final_features)
    history.append(final_ratio.item())

    return "".join(current_seq), history, initial_gradients


def plot_optimization_curve(history: list[float], task_id: str):
    """
    Wyrysowuje i zapisuje wykres liniowy.
    Wykazuje przyrost predykcji po każdej K-elementowej transakcji mutacyjnej.
    Generuje wymagany przez instrukcję artefakt do prezentacji.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, marker='o', linestyle='-', color='b')
    plt.title(f"Optimization Curve - {task_id}")
    plt.xlabel("Liczba iteracji (wiele mutacji na krok)")
    plt.ylabel("Predicted RNA/DNA Ratio")
    plt.grid(True)
    plt.tight_layout()
    safe_name = "".join([c if c.isalnum() else "_" for c in task_id])
    plt.savefig(f"{safe_name}_curve.png")
    plt.close()


def plot_saliency_map(gradients: np.ndarray, task_id: str):
    """
    Renderuje mapę cieplną (Heatmap) wyciągniętą bezpośrednio z silnika wnioskowania modelu.
    Wykazuje na 2 osiach (Pozycja oraz Nukleotyd), które komórki macierzy powodowały
    najwyższą aktywację głowicy. Potwierdza zjawisko wyłapywania biologicznych elementów regulatorowych.
    """
    plt.figure(figsize=(15, 3))
    sns.heatmap(gradients[:, :50], cmap="coolwarm", center=0, yticklabels=['A', 'C', 'G', 'T'])
    plt.title(f"Saliency Map (Initial Gradients) - {task_id} (Pierwsze 50 bp)")
    plt.xlabel("Pozycja w sekwencji")
    plt.ylabel("Nukleotyd")
    plt.tight_layout()
    safe_name = "".join([c if c.isalnum() else "_" for c in task_id])
    plt.savefig(f"{safe_name}_saliency.png")
    plt.close()


def main():
    """
    Kontroler przepływu uruchamiający optymalizację według narzuconych limitów.
    1. Ustala powtarzalne środowisko.
    2. Replikuje i wycisza Dropout i BatchNorm ładując stan wyuczony z `best_checkpoint.pt`.
    3. Ekstrahuje uszkodzone wzmacniacze i sekwencje niskiej aktywności.
    4. Rozsyła zadania do algorytmu podając twarde limity z PDF (40, 16, 20 i konfigurowalny limit dla B).
    5. Buduje tabelę z wynikami wymaganą do oddania jako TSV.
    """
    set_seed(3)
    device = select_device()
    print(f"Inicjalizacja na: {device}")

    model_path = Path("best_checkpoint.pt")
    if not model_path.exists():
        print("BŁĄD: Brak pliku best_checkpoint.pt.")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ResidualDilatedMultitaskCnn(
        dropout=float(checkpoint.get("dropout", 0.20)),
        channels=int(checkpoint.get("channels", 96)),
        dilations=tuple(int(d) for d in checkpoint.get("dilations", [1, 2, 4, 8, 16])),
        dense_units=int(checkpoint.get("dense_units", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    subtask_a_seqs = read_fasta("subtaskA.fa")
    subtask_b_seqs = read_fasta("subtaskB.fa")

    if not subtask_a_seqs and not subtask_b_seqs:
        print("BŁĄD: Brak sekwencji z plików FASTA.")
        sys.exit(1)

    tasks = []

    # Limity dla poszczególnych zadań
    subtask_a_limits = [40, 16, 20]
    LIMIT_ZADANIA_B = 200

    for idx, (seq_id, seq) in enumerate(subtask_a_seqs.items()):
        limit = subtask_a_limits[idx] if idx < len(subtask_a_limits) else 40
        tasks.append({"id": f"A_{seq_id}", "sequence": seq, "limit": limit})

    for seq_id, seq in subtask_b_seqs.items():
        tasks.append({"id": f"B_{seq_id}", "sequence": seq, "limit": LIMIT_ZADANIA_B})

    results = []
    K_MUTATIONS = 5

    for task in tasks:
        limit_str = str(task['limit']) if task['limit'] is not None else "BRAK LIMITU"
        print(f"\nOptymalizacja: {task['id']} | Limit: {limit_str} | Top-K: {K_MUTATIONS}")

        opt_seq, history, initial_grads = optimize_by_gradient(
            model, task['sequence'], device, max_mutations=task['limit'], k_mutations=K_MUTATIONS
        )

        print(f"  Wynik początkowy: {history[0]:.4f} \n           Końcowy: {history[-1]:.4f}")

        results.append({
            "id": task['id'],
            "new_sequence": opt_seq,
            "predicted_rna_dna_ratio": history[-1]
        })

        plot_optimization_curve(history, task['id'])
        plot_saliency_map(initial_grads, task['id'])

    if results:
        pd.DataFrame(results).to_csv("optimized_sequences.tsv", sep="\t", index=False)
        print("\nZapisano pliki wynikowe.")


if __name__ == "__main__":
    main()