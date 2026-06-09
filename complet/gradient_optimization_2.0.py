#!/usr/bin/env python3
"""
Skrypt do inżynierii sekwencji DNA wykorzystujący Atrybucję Gradientową (Top-K).
Zapewnia powtarzalność wyników dzięki globalnemu ustawieniu ziarna losowości (seed).
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

SEQUENCE_LENGTH = 230


def set_seed(seed: int = 3) -> None:
    """
    Ustawia ziarno losowości dla wszystkich wykorzystywanych bibliotek.
    Gwarantuje powtarzalność wyników przy tym samym modelu i danych wejściowych.
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
    - Squeeze: Kompresuje informacje z całej sekwencji (AdaptiveAvgPool1d) do jednego wektora.
    - Excitation: Dwie warstwy konwolucyjne z ReLU i Sigmoid obliczają wagi (0–1) dla każdego kanału.
    Mnoży wejście przez te wagi, wyciszając kanały stanowiące szum.
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
        return x * self.se(x)


class ResidualDilatedBlock(nn.Module):
    """
    Rezydualny blok ekstrakcji cech z dylatacją.
    Dylatacja pozwala sieci widzieć coraz szersze fragmenty sekwencji DNA,
    wychwytując współdziałanie odległych motywów bez zwiększania liczby wag.
    Zawiera skip-connection zapobiegające degradacji gradientów.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
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
        return self.activation(inputs + self.block(inputs))


class ResidualDilatedMultitaskCnn(nn.Module):
    """
    Architektura głównej sieci CNN.
    - stem: Warstwa wejściowa lokalizująca podstawowe k-mery.
    - residual_blocks: 5 bloków z wykładniczo rosnącą dylatacją.
    - pooling: Agreguje macierz przestrzenną (średnia + maksimum).
    - multitask heads: Regresja (RNA/DNA ratio) i klasyfikacja (is_active).
    """

    def __init__(
        self,
        dropout: float = 0.20,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dense_units: int = 128,
    ) -> None:
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
        features = self.residual_blocks(self.stem(inputs))
        max_pooled = torch.amax(features, dim=2)
        avg_pooled = torch.mean(features, dim=2)
        shared = self.shared_dense(torch.cat((max_pooled, avg_pooled), dim=1))
        return self.regression_head(shared), self.classification_head(shared)


def one_hot_encode(sequences: list[str]) -> np.ndarray:
    """
    Zamienia sekwencje DNA na macierze [N × 4 × 230] zrozumiałe dla sieci.
    A=0, C=1, G=2, T=3. Na każdej pozycji dokładnie jedna komórka ma wartość 1.
    """
    features = np.zeros((len(sequences), 4, SEQUENCE_LENGTH), dtype=np.float32)
    channel_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row_index, sequence in enumerate(sequences):
        for col_index, base in enumerate(sequence.upper()):
            if base in channel_by_base:
                features[row_index, channel_by_base[base], col_index] = 1.0
    return features


def select_device() -> torch.device:
    """Wybiera GPU jeśli dostępne, w przeciwnym razie CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta(file_path: str | Path) -> dict[str, str]:
    """
    Wczytuje plik FASTA i zwraca słownik {nazwa_sekwencji: sekwencja}.
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


def optimize_by_gradient(
    model: nn.Module,
    start_seq: str,
    device: torch.device,
    max_mutations: int | None = None,
    k_mutations: int = 5,
) -> tuple[str, list[float], np.ndarray]:
    """
    Optymalizacja sekwencji DNA metodą Atrybucji Gradientowej (Top-K).

    1. Koduje sekwencję One-Hot z requires_grad=True.
    2. Forward pass → pred_ratio.
    3. Backward pass → gradienty względem wejścia.
    4. Dla każdej pozycji oblicza improvement = grad[nowa_baza] - grad[stara_baza].
    5. Aplikuje Top-K najlepszych mutacji na unikalnych pozycjach.
    6. Powtarza aż do wyczerpania budżetu, plateau lub wykrycia cyklu.

    Budżet (max_mutations) liczy unikalne pozycje — nadpisanie tej samej pozycji
    nie zużywa dodatkowego budżetu.
    """
    bases = ["A", "C", "G", "T"]
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    current_seq = list(start_seq.upper())
    mutations_done = 0
    history: list[float] = []
    initial_gradients: np.ndarray | None = None
    mutation_log: list[tuple[int, int, int, int]] = []
    step = 0
    ever_mutated: set[int] = set()
    seen_sequences: set[int] = {hash("".join(current_seq))}

    while True:
        if max_mutations is not None and mutations_done >= max_mutations:
            break

        features_np = one_hot_encode(["".join(current_seq)])
        features = torch.from_numpy(features_np.copy()).to(device).requires_grad_(True)

        pred_ratio, _ = model(features)
        model.zero_grad()
        pred_ratio.backward()

        gradients = features.grad.cpu().numpy()[0]
        if initial_gradients is None:
            initial_gradients = gradients.copy()

        possible_mutations: list[tuple[float, int, str]] = []

        for pos in range(len(current_seq)):
            current_base = current_seq[pos]
            if current_base not in base_to_idx:
                continue
            current_idx = base_to_idx[current_base]

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
        mutated_positions: set[int] = set()

        for imp, pos, base in possible_mutations:
            if pos not in mutated_positions:
                old_base = current_seq[pos]
                old_idx = base_to_idx[old_base]
                new_idx = base_to_idx[base]

                current_seq[pos] = base
                mutation_log.append((step, pos, old_idx, new_idx))
                mutated_positions.add(pos)

                if pos not in ever_mutated:
                    mutations_done += 1
                    ever_mutated.add(pos)

                applied_this_step += 1

                if applied_this_step >= k_mutations or (
                    max_mutations is not None and mutations_done >= max_mutations
                ):
                    break

        if applied_this_step == 0:
            break

        seq_hash = hash("".join(current_seq))
        if seq_hash in seen_sequences:
            break
        seen_sequences.add(seq_hash)

        step += 1

        with torch.no_grad():
            step_features = torch.from_numpy(
                one_hot_encode(["".join(current_seq)])
            ).to(device)
            step_ratio, _ = model(step_features)
            history.append(step_ratio.item())

    with torch.no_grad():
        final_features = torch.from_numpy(
            one_hot_encode(["".join(current_seq)])
        ).to(device)
        final_ratio, _ = model(final_features)

    if not history:
        history.append(final_ratio.item())

    if initial_gradients is None:
        initial_gradients = np.zeros((4, SEQUENCE_LENGTH), dtype=np.float32)

    mutations_array = (
        np.array(mutation_log, dtype=int)
        if mutation_log
        else np.empty((0, 4), dtype=int)
    )
    return "".join(current_seq), history, mutations_array


def main() -> None:
    """
    Uruchamia optymalizację dla wszystkich sekwencji z subtaskA.fa i subtaskB.fa.
    Zapisuje wyniki do optimized_sequences.tsv oraz {task_id}_mutations.npy.
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
    K_MUTATIONS = 5

    for task in tasks:
        limit_str = str(task["limit"]) if task["limit"] is not None else "BRAK LIMITU"
        print(f"\nOptymalizacja: {task['id']} | Limit: {limit_str} | Top-K: {K_MUTATIONS}")

        opt_seq, history, mutations_array = optimize_by_gradient(
            model, task["sequence"], device,
            max_mutations=task["limit"], k_mutations=K_MUTATIONS,
        )

        print(f"  Wynik końcowy: {history[-1]:.4f}")

        results.append({
            "id": task["id"],
            "new_sequence": opt_seq,
            "predicted_rna_dna_ratio": history[-1],
        })

        sn = "".join(c if c.isalnum() else "_" for c in task["id"])
        np.save(f"{sn}_mutations.npy", mutations_array)

    if results:
        pd.DataFrame(results).to_csv("optimized_sequences.tsv", sep="\t", index=False)
        print("\nZapisano pliki wynikowe.")


if __name__ == "__main__":
    main()