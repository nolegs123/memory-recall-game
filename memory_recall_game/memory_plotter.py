from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_csv() -> Path | None:
    """Use a CSV next to the script if possible, otherwise ask the user to choose one."""
    folder = Path(__file__).resolve().parent

    preferred = folder / "memory_results(1).csv"
    if preferred.exists():
        return preferred

    candidates = sorted(folder.glob("memory_results*.csv"))
    if candidates:
        return candidates[0]

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Choose memory results CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def load_and_summarize(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = {"position", "remembered"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    # Accept True/False, 1/0, and common text forms.
    if df["remembered"].dtype != bool:
        normalized = df["remembered"].astype(str).str.strip().str.lower()
        df["remembered"] = normalized.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        if df["remembered"].isna().any():
            raise ValueError("The 'remembered' column contains values I cannot interpret.")

    group_col = "mode" if "mode" in df.columns else None

    if group_col:
        summary = (
            df.groupby(["position", group_col], as_index=False)
            .agg(
                remembered_count=("remembered", "sum"),
                attempts=("remembered", "size"),
            )
        )
    else:
        summary = (
            df.groupby("position", as_index=False)
            .agg(
                remembered_count=("remembered", "sum"),
                attempts=("remembered", "size"),
            )
        )
        summary["mode"] = "All runs"
        group_col = "mode"

    summary["remembered_percent"] = (
        summary["remembered_count"] / summary["attempts"] * 100
    )

    return df, summary


def pretty_mode_name(mode: str) -> str:
    names = {
        "free_recall_fast": "Fast (1 s/word)",
        "free_recall_slow": "Slow (3 s/word)",
    }
    return names.get(str(mode), str(mode).replace("_", " ").title())


def plot_counts(summary: pd.DataFrame, source_name: str) -> None:
    modes = list(summary["mode"].drop_duplicates())
    positions = sorted(summary["position"].drop_duplicates())
    x = np.arange(len(positions))

    n_modes = max(len(modes), 1)
    total_width = 0.78
    bar_width = total_width / n_modes

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, mode in enumerate(modes):
        mode_data = (
            summary[summary["mode"] == mode]
            .set_index("position")
            .reindex(positions)
        )

        counts = mode_data["remembered_count"].fillna(0).to_numpy()
        attempts = mode_data["attempts"].fillna(0).to_numpy()
        offset = (i - (n_modes - 1) / 2) * bar_width

        bars = ax.bar(
            x + offset,
            counts,
            width=bar_width * 0.94,
            label=pretty_mode_name(mode),
        )

        for bar, count, total in zip(bars, counts, attempts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{int(count)}/{int(total)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    max_attempts = int(summary["attempts"].max()) if len(summary) else 1
    ax.set_title("Words remembered by serial position")
    ax.set_xlabel("Word position in the sequence")
    ax.set_ylabel("Number of times remembered")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_ylim(0, max_attempts + max(2, max_attempts * 0.12))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(source_name, fontsize=9, y=0.995)
    fig.tight_layout()
    plt.show()


def main() -> None:
    path = find_csv()
    if path is None:
        return

    try:
        _, summary = load_and_summarize(path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not plot file", str(exc))
        root.destroy()
        return

    print("\nRemembered by position:\n")
    printable = summary.copy()
    printable["mode"] = printable["mode"].map(pretty_mode_name)
    printable["remembered_percent"] = printable["remembered_percent"].round(1)
    print(printable.to_string(index=False))

    plot_counts(summary, path.name)


if __name__ == "__main__":
    main()
