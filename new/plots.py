#!/usr/bin/env python3
"""
Skrypt wizualizacyjny dla wyników dna_optimizer.py.

Wymaga plików wygenerowanych przez dna_optimizer.py:
  - optimized_sequences.tsv
  - {task_id}_mutations.npy  (log mutacji: [krok, pozycja, stara_baza_idx, nowa_baza_idx])
  - subtaskA.fa, subtaskB.fa (oryginalne sekwencje)

Uruchomienie:
  1. python dna_optimizer.py
  2. python visualize_results.py

Wyjście: plot4_{task_id}_comparison.png dla każdego zadania.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Stałe ─────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 230
FIGURE_DPI = 150
OUTPUT_DIR = Path(".")

# ── Helpers ───────────────────────────────────────────────────────────────────


def safe_name(task_id: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in task_id)


def read_fasta(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
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


# ── Wykres: porównanie sekwencji przed i po ───────────────────────────────────


def plot_sequence_comparison(
    task_id: str,
    original_seq: str,
    optimized_seq: str,
    mutations: np.ndarray,
) -> None:
    """Dwa paski: oryginalna i zoptymalizowana sekwencja.

    Zmienione pozycje podświetlone kolorem bazy; niezmienione szare.
    """
    changed_positions = set(int(r[1]) for r in mutations)

    fig, axes = plt.subplots(
        2, 1, figsize=(40, 8), gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.subplots_adjust(hspace=0.35)

    base_colors = {"A": "#2ecc71", "C": "#3498db", "G": "#e67e22", "T": "#e74c3c"}
    neutral_color = "#ECECEC"

    SEQ_FONT = 18
    LABEL_FONT = 30
    TITLE_FONT = 30

    for ax_idx, (seq, label) in enumerate(
        [(original_seq, "Original"), (optimized_seq, "Mutated")]
    ):
        ax = axes[ax_idx]
        ax.set_xlim(0, SEQUENCE_LENGTH)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=LABEL_FONT, rotation=90, labelpad=60, va="center")

        if ax_idx == 0:
            ax.set_xticks([])
        else:
            tick_step = 10 if SEQUENCE_LENGTH <= 150 else 20
            tick_positions = list(range(0, SEQUENCE_LENGTH, tick_step))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(p) for p in tick_positions], fontsize=LABEL_FONT - 4)
            ax.set_xlabel("Position (bp)", fontsize=LABEL_FONT)

        for pos in range(min(SEQUENCE_LENGTH, len(seq))):
            base = seq[pos]
            color = base_colors.get(base, "#888888") if pos in changed_positions else neutral_color

            rect = mpatches.FancyBboxPatch(
                (pos + 0.05, 0.1), 0.90, 0.80,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
            )
            ax.add_patch(rect)

            if pos in changed_positions:
                ax.text(
                    pos + 0.5, 0.5, base,
                    ha="center", va="center",
                    fontsize=SEQ_FONT, color="white",
                    fontweight="bold", fontfamily="monospace",
                )

    n_changed = len(changed_positions)

    legend_patches = [mpatches.Patch(color=c, label=b) for b, c in base_colors.items()]
    legend_patches.append(mpatches.Patch(color=neutral_color, label="unchanged"))
    axes[0].legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=LABEL_FONT - 2,
        ncol=5,
        frameon=False,
    )

    axes[0].set_title(
        f"Sequence comparison — {task_id}  ({n_changed} mutated bp along {SEQUENCE_LENGTH} bp)",
        fontsize=TITLE_FONT,
        fontweight="bold",
        pad=80,
    )

    out_path = OUTPUT_DIR / f"plot4_{safe_name(task_id)}_comparison.png"
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=== Wizualizacja wyników dna_optimizer.py ===\n")

    tsv_path = OUTPUT_DIR / "optimized_sequences.tsv"
    if not tsv_path.exists():
        print(f"BŁĄD: Brak pliku {tsv_path}. Uruchom najpierw dna_optimizer.py.")
        return

    tsv = pd.read_csv(tsv_path, sep="\t")
    print(f"Wczytano {len(tsv)} zadań z {tsv_path}\n")

    # Wczytaj oryginalne sekwencje
    original_seqs: dict[str, str] = {}
    for fa_file, prefix in [("subtaskA.fa", "A"), ("subtaskB.fa", "B")]:
        for seq_id, seq in read_fasta(OUTPUT_DIR / fa_file).items():
            original_seqs[f"{prefix}_{seq_id}"] = seq

    print("Generowanie wykresów:")

    for _, row in tsv.iterrows():
        task_id = row["id"]
        sn = safe_name(task_id)
        mutations_path = OUTPUT_DIR / f"{sn}_mutations.npy"

        if not mutations_path.exists():
            print(f"  [BRAK] {task_id}: brak pliku {mutations_path}")
            continue

        mutations = np.load(mutations_path)
        original_seq = original_seqs.get(task_id, "N" * SEQUENCE_LENGTH)
        optimized_seq = row["new_sequence"]

        plot_sequence_comparison(task_id, original_seq, optimized_seq, mutations)

    print("\nGotowe.")


if __name__ == "__main__":
    main()