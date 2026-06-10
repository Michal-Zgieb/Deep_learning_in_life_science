#!/usr/bin/env python3
"""
Skrypt do zaawansowanej optymalizacji sekwencji DNA za pomocą przeszukiwania wiązkowego.
Wykorzystuje atrybucję gradientową jako filtr wstępny oraz rzeczywiste predykcje modelu
jako twarde kryterium selekcji, co eliminuje błędy wynikające z nieliniowości sieci CNN.
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
    Ustawia stałe ziarno losowości dla generatorów PyTorch, NumPy oraz modułu random.
    Gwarantuje identyczny przebieg przeszukiwania wiązkowego przy każdym uruchomieniu.

    Args:
        seed (int): Wartość inicjalizująca dla generatorów liczb pseudolosowych.
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
    Moduł uwagi kanałowej Squeeze-and-Excitation dla sygnałów jednowymiarowych (1D).

    Dynamicznie waży istotność poszczególnych kanałów cech wyekstrahowanych
    przez warstwy konwolucyjne. Składa się z etapu 'Squeeze' (agregacja globalnego
    kontekstu za pomocą AdaptiveAvgPool1d) oraz 'Excitation' (dwuwarstwowy perceptron
    redukujący i odtwarzający wymiar kanałów z aktywacją Sigmoid). Wynikowy wektor
    wag służy do skalowania map cech, co pozwala wyciszyć szum tła sekwencji.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        Inicjalizuje warstwy liniowe (zrealizowane jako Conv1d z kernelem 1) dla bloku SE.

        Args:
            channels (int): Liczba kanałów wejściowych mapy cech.
            reduction (int): Współczynnik redukcji wymiarowości w warstwie ukrytej perceptrona.
        """
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
        """
        Przetwarza tensor wejściowy i nakłada wyliczone wagi uwagi kanałowej.

        Args:
            x (torch.Tensor): Tensor cech o kształcie [Batch, Channels, Length].

        Returns:
            torch.Tensor: Przeskalowany tensor wejściowy o identycznym kształcie.
        """
        return x * self.se(x)


class ResidualDilatedBlock(nn.Module):
    """
    Rezydualny blok konwolucyjny wykorzystujący sploty z dylatacją (rozszerzeniem).

    Dylatacja pozwala na wykładnicze zwiększanie efektywnego pola widzenia (receptive field)
    filtrów bez zwiększania liczby parametrów sieci. Umożliwia to modelowi wykrywanie
    odległych zależności przestrzennych między motywami regulatorowymi w DNA. Blok zawiera
    dwie warstwy splotowe, normalizację Batch Normalization, aktywację GELU, Dropout,
    blok uwagi SEBlock oraz połączenie skrócone (skip connection) zapobiegające zanikaniu gradientu.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        """
        Inicjalizuje warstwy konwolucyjne i normalizacyjne z uwzględnieniem zadanego kroku dylatacji.

        Args:
            channels (int): Liczba kanałów przetwarzanych wewnątrz bloku.
            dilation (int): Wartość dylatacji (rozszerzenia) dla warstw konwolucyjnych.
            dropout (float): Prawdopodobieństwo wyzerowania aktywacji w warstwie regularyzacyjnej.
        """
        super().__init__()
        kernel_size = 5
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            SEBlock(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Realizuje połączenie rezydualne, sumując wejście bezpośrednie z przetworzonym przez blok.

        Args:
            inputs (torch.Tensor): Tensor wejściowy o kształcie [Batch, Channels, Length].

        Returns:
            torch.Tensor: Wynik sumowania po przejściu przez nieliniowość GELU.
        """
        return self.activation(inputs + self.block(inputs))


class ResidualDilatedMultitaskCnn(nn.Module):
    """
    Główna wielozadaniowa sieć konwolucyjna do predykcji aktywności sekwencji regulatorowych.

    - Warstwa wejściowa (stem): Ekstrahuje pierwotne cechy lokalnych k-merów za pomocą konwolucji.
    - Blok rezydualny: Przetwarza sygnał przez sekwencję bloków o rosnącej dylatacji (1, 2, 4, 8, 16).
    - Agregacja przestrzenna: Łączy globalny pooling maksymalny i średni w celu zachowania informacji
      o obecności i intensywności motywów biologicznych.
    - Głowice wyjściowe: Równolegle realizuje regresję liniową parametru rna_dna_ratio oraz klasyfikację aktywności.
    """

    def __init__(
            self,
            dropout: float = 0.20,
            channels: int = 96,
            dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
            dense_units: int = 128,
    ) -> None:
        """
        Konstruuje pełne warstwy sieci neuronowej na podstawie zadanych hiperparametrów.

        Args:
            dropout (float): Współczynnik odrzucenia dla warstw Dropout.
            channels (int): Liczba kanałów ukrytych w blokach konwolucyjnych.
            dilations (tuple[int, ...]): Sekwencja kroków dylatacji dla kolejnych bloków.
            dense_units (int): Liczba neuronów w wspólnej warstwie w pełni połączonej.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualDilatedBlock(channels, dilation, dropout) for dilation in dilations]
        )
        self.shared_dense = nn.Sequential(
            nn.Linear(channels * 2, dense_units),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(dense_units, 1)
        self.classification_head = nn.Linear(dense_units, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wykonuje pełne przejście w przód (Forward Pass) przez sieć wielozadaniową.

        Args:
            inputs (torch.Tensor): Tensor z zakodowanymi one-hot sekwencjami o kształcie [Batch, 4, 230].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Para tensorów wyjściowych: (wynik_regresji, logity_klasyfikacji).
        """
        features = self.residual_blocks(self.stem(inputs))
        max_pooled = torch.amax(features, dim=2)
        avg_pooled = torch.mean(features, dim=2)
        shared = self.shared_dense(torch.cat((max_pooled, avg_pooled), dim=1))
        return self.regression_head(shared), self.classification_head(shared)


def one_hot_encode(sequences: list[str]) -> np.ndarray:
    """
    Dokonuje transformacji tekstowych sekwencji nukleotydowych na postać macierzy binarnych.

    Mapuje zasady azotowe na konkretne indeksy kanałów: A -> 0, C -> 1, G -> 2, T -> 3.

    Args:
        sequences (list[str]): Lista ciągów tekstowych reprezentujących sekwencje DNA o stałej długości.

    Returns:
        np.ndarray: Tablica NumPy o kształcie [Liczba_sekwencji, 4, 230] i typie float32.
    """
    features = np.zeros((len(sequences), 4, SEQUENCE_LENGTH), dtype=np.float32)
    channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row_index, sequence in enumerate(sequences):
        for col_index, base in enumerate(sequence.upper()):
            if base in channel_by_base:
                features[row_index, channel_by_base[base], col_index] = 1.0
    return features


def select_device() -> torch.device:
    """
    Weryfikuje konfigurację sprzętową stacji roboczej pod kątem akceleracji obliczeń.

    Returns:
        torch.device: Obiekt wskazujący na akcelerator 'cuda' (jeśli dostępny) lub procesor 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta(file_path: str | Path) -> dict[str, str]:
    """
    Parsuje pliki wejściowe zapisane w standardowym formacie bioinformatycznym FASTA.



    Args:
        file_path (str | Path): Ścieżka do docelowego pliku FASTA.

    Returns:
        dict[str, str]: Słownik mapujący identyfikatory sekwencji (klucze) na surowe ciągi nukleotydów (wartości).
    """
    path = Path(file_path)
    if not path.exists():
        print(f"BŁĄD: Nie znaleziono pliku {file_path}!")
        return {}
    sequences: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        current_id: str | None = None
        current_seq: list[str] = []
        for line in f:
            line = line.strip()
            if not line: continue
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


def optimize_by_beam_search(
        model: nn.Module,
        start_seq: str,
        device: torch.device,
        max_mutations: int | None = None,
        beam_width: int = 5,
        top_k_gradients: int = 20
) -> tuple[str, list[float], np.ndarray]:
    """
    Optymalizuje sekwencję DNA za pomocą algorytmu Beam Search wspomaganego gradientowo.

    Zasada działania opiera się na eliminacji nieliniowych błędów aproksymacji gradientu:
    1. Dla każdej z `beam_width` (25) aktualnie najlepszych sekwencji wyliczany jest gradient.
    2. Na bazie różnic gradientów generuje się wyłącznie `top_k_gradients` (100) najbardziej obiecujących mutacji punktowych.
    3. Wszystkie unikalne warianty kandydujące (max 100 na krok) są fizycznie tworzone i oceniane
       poprzez twardy Forward Pass modelu w celu pobrania ich rzeczywistego `rna_dna_ratio`.
    4. Pula kandydatów jest sortowana, a 25 wariantów z najwyższym realnym wynikiem tworzy nową wiązkę (Beam) dla kolejnego kroku.
    5. Proces kończy się w momencie osiągnięcia limitu mutacji unikalnych pozycji lub po wykryciu plateau.

    Args:
        model (nn.Module): Wytrenowana sieć neuronowa służąca jako środowisko symulacyjne.
        start_seq (str): Pierwotna sekwencja nukleotydowa stanowiąca punkt początkowy inżynierii.
        device (torch.device): Urządzenie obliczeniowe (CPU/GPU).
        max_mutations (int | None): Maksymalny budżet modyfikacji unikalnych pozycji (odległość Hamminga).
        beam_width (int): Szerokość wiązki (liczba równolegle utrzymywanych najlepszych ścieżek).
        top_k_gradients (int): Liczba mutacji generowanych przez gradient dla pojedynczej sekwencji z wiązki.

    Returns:
        tuple[str, list[float], np.ndarray]:
            - zoptymalizowana sekwencja końcowa (str),
            - historia najlepszych rzeczywistych wyników rna_dna_ratio na krok (list[float]),
            - początkowa macierz gradientów wejściowych 4x230 dla sekwencji startowej (np.ndarray).
    """
    bases = ["A", "C", "G", "T"]
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    init_feat_np = one_hot_encode([start_seq.upper()])
    init_feat = torch.from_numpy(init_feat_np).to(device).requires_grad_(True)
    init_pred, _ = model(init_feat)

    model.zero_grad()
    init_pred.backward()
    initial_gradients = init_feat.grad.cpu().numpy()[0].copy()
    initial_score = init_pred.item()


    beam = [(initial_score, list(start_seq.upper()), 0, set())]
    seen_sequences = {hash(start_seq.upper())}

    best_overall_seq = list(start_seq.upper())
    best_overall_score = initial_score

    history = [initial_score]
    patience_counter = 0

    while True:
        candidates_seqs = []
        candidates_meta = []

        for b_score, b_seq, b_mut_count, b_mut_pos in beam:
            if max_mutations is not None and b_mut_count >= max_mutations:
                continue

            feat_np = one_hot_encode(["".join(b_seq)])
            feat = torch.from_numpy(feat_np).to(device).requires_grad_(True)
            p, _ = model(feat)
            model.zero_grad()
            p.backward()
            grad = feat.grad.cpu().numpy()[0]

            possible_mutations = []
            for pos in range(len(b_seq)):
                if pos in b_mut_pos: continue

                curr_base = b_seq[pos]
                curr_idx = base_to_idx.get(curr_base, -1)
                if curr_idx == -1: continue

                for cand_base in bases:
                    if cand_base == curr_base: continue
                    cand_idx = base_to_idx[cand_base]
                    improvement = grad[cand_idx, pos] - grad[curr_idx, pos]

                    if improvement > 1e-6:
                        possible_mutations.append((improvement, pos, curr_base, cand_base))

            possible_mutations.sort(reverse=True, key=lambda x: x[0])

            for imp, pos, old_base, new_base in possible_mutations[:top_k_gradients]:
                new_seq = b_seq.copy()
                new_seq[pos] = new_base
                new_seq_str = "".join(new_seq)

                h = hash(new_seq_str)
                if h not in seen_sequences:
                    seen_sequences.add(h)
                    new_mut_pos = b_mut_pos.copy()
                    new_mut_pos.add(pos)

                    candidates_seqs.append(new_seq_str)
                    candidates_meta.append((new_mut_pos, b_mut_count + 1))

        if not candidates_seqs:
            break

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(candidates_seqs), 512):
                batch = candidates_seqs[i:i + 512]
                batch_feat = torch.from_numpy(one_hot_encode(batch)).to(device)
                p_batch, _ = model(batch_feat)
                all_preds.extend(p_batch.cpu().numpy().flatten())

        scored_candidates = []
        for i in range(len(candidates_seqs)):
            scored_candidates.append((
                all_preds[i],
                list(candidates_seqs[i]),
                candidates_meta[i][1],
                candidates_meta[i][0]
            ))

        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        beam = scored_candidates[:beam_width]

        step_best_score = beam[0][0]
        history.append(step_best_score)

        if step_best_score > best_overall_score:
            best_overall_score = step_best_score
            best_overall_seq = beam[0][1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 25:
            break
        if max_mutations is not None and all(b[2] >= max_mutations for b in beam):
            break

    return "".join(best_overall_seq), history, initial_gradients


def plot_optimization_curve(history: list[float], task_id: str):
    """
    Wyrysowuje i eksportuje wykres krzywej optymalizacji dla danego podzadania.

    Wykres przedstawia wzrost rzeczywistego, zweryfikowanego parametru rna_dna_ratio
    na osi Y w odniesieniu do kolejnych kroków (rund) algorytmu przeszukiwania wiązkowego na osi X.
    Spełnia formalny wymóg dokumentacji projektu: 'The Optimization Curve'.

    Args:
        history (list[float]): Lista rzeczywistych wartości rna_dna_ratio uzyskanych w kolejnych krokach.
        task_id (str): Identyfikator zadania używany jako nazwa pliku i element nagłówka wykresu.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, marker='o', linestyle='-', color='b')
    plt.title(f"Optimization Curve - {task_id}")
    plt.xlabel("Iteracja (Krok przeszukiwania wiązkowego)")
    plt.ylabel("Predicted RNA/DNA Ratio")
    plt.grid(True)
    plt.tight_layout()
    safe_name = "".join([c if c.isalnum() else "_" for c in task_id])
    plt.savefig(f"{safe_name}_curve.png")
    plt.close()


def plot_saliency_map(gradients: np.ndarray, task_id: str):
    """
    Generuje i eksportuje pełnowymiarową mapę istotności (Saliency Map) dla całej sekwencji.

    Wizualizuje początkową macierz gradientów 4x230 jako dwuwymiarową heatmapę.
    Wskazuje pozycje i konkretne nukleotydy, na które model wykazuje najwyższą wrażliwość,
    co służy jako matematyczny dowód na identyfikację biologicznych motywów regulatorowych.

    Args:
        gradients (np.ndarray): Macierz gradientów o kształcie [4, 230] pobrana dla pierwotnej sekwencji.
        task_id (str): Identyfikator zadania stosowany do parametryzacji zapisu grafiki końcowej.
    """
    plt.figure(figsize=(25, 3))
    sns.heatmap(gradients, cmap="coolwarm", center=0, yticklabels=['A', 'C', 'G', 'T'])
    plt.title(f"Saliency Map (Initial Gradients) - {task_id} (Pełna sekwencja 230 bp)")
    plt.xlabel("Pozycja w sekwencji")
    plt.ylabel("Nukleotyd")
    plt.tight_layout()
    safe_name = "".join([c if c.isalnum() else "_" for c in task_id])
    plt.savefig(f"{safe_name}_saliency.png")
    plt.close()


def main() -> None:
    """
    Główna funkcja sterująca potokiem przetwarzania i inżynierii sekwencji DNA.

    1. Ustala stabilne środowisko losowości (seed=3).
    2. Odczytuje konfigurację i wagi sieci neuronowej z pliku checkpointu 'best_checkpoint.pt'.
    3. Przełącza sieć w tryb ewaluacji (`model.eval()`), dezaktywując warstwy regularyzacji.
    4. Ładuje sekwencje startowe z plików FASTA (`subtaskA.fa` i `subtaskB.fa`).
    5. Konfiguruje twarde limity modyfikacji z tabeli projektowej (40, 16, 20) oraz limit dla zadania B (200).
    6. Wykonuje zaawansowany proces Beam Search, generując pliki raportowe .tsv i wykresy .png.
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

    tasks: list[dict] = []
    subtask_a_limits = [40, 16, 20]
    LIMIT_ZADANIA_B = 200

    for idx, (seq_id, seq) in enumerate(subtask_a_seqs.items()):
        limit = subtask_a_limits[idx] if idx < len(subtask_a_limits) else 40
        tasks.append({"id": f"A_{seq_id}", "sequence": seq, "limit": limit})

    for seq_id, seq in subtask_b_seqs.items():
        tasks.append({"id": f"B_{seq_id}", "sequence": seq, "limit": LIMIT_ZADANIA_B})

    results = []
    for task in tasks:
        limit_str = str(task["limit"]) if task["limit"] is not None else "BRAK LIMITU"
        print(f"\nOptymalizacja: {task['id']} | Limit: {limit_str} | Beam: 25 | Top_K_Grad: 100")


        opt_seq, history, initial_grads = optimize_by_beam_search(
            model, task["sequence"], device,
            max_mutations=task["limit"], beam_width=25, top_k_gradients=100
        )

        print(f"  Wynik początkowy: {history[0]:.4f}")
        print(f"  Wynik końcowy:    {history[-1]:.4f}")

        results.append({
            "id": task["id"],
            "new_sequence": opt_seq,
            "predicted_rna_dna_ratio": history[-1],
        })

        plot_optimization_curve(history, task['id'])
        plot_saliency_map(initial_grads, task['id'])



    if results:
        pd.DataFrame(results).to_csv("optimized_sequences.tsv", sep="\t", index=False)
        print("\nZapisano pliki wynikowe (TSV, PNG).")


if __name__ == "__main__":
    main()