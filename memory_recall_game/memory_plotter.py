import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_path(value: str) -> str:
    return value.strip().strip('"').strip("'")


def find_csv(argument: str | None = None) -> Path | None:
    if argument:
        return Path(clean_path(argument))

    folder = Path(__file__).resolve().parent

    preferred = folder / "memory_results(1).csv"
    if preferred.exists():
        return preferred

    candidates = sorted(folder.glob("memory_results*.csv"))
    if candidates:
        print(f"Using: {candidates[0]}")
        return candidates[0]

    entered = input("CSV file path: ").strip()
    return Path(clean_path(entered)) if entered else None


def load_and_summarize(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = {"position", "remembered"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

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


def plot_counts(summary: pd.DataFrame, source_name: str, output_path: Path) -> None:
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
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a serial-position memory plot from a CSV file."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="CSV file to plot. If omitted, the program auto-detects one or asks for a path."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Default: <csv name>_plot.png next to the CSV."
    )
    args = parser.parse_args()

    path = find_csv(args.csv_file)
    if path is None:
        print("No CSV file supplied.")
        return 1

    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    try:
        _, summary = load_and_summarize(path)
    except Exception as exc:
        print(f"Could not plot file: {exc}")
        return 1

    print("\nRemembered by position:\n")
    printable = summary.copy()
    printable["mode"] = printable["mode"].map(pretty_mode_name)
    printable["remembered_percent"] = printable["remembered_percent"].round(1)
    print(printable.to_string(index=False))

    if args.output:
        output_path = Path(clean_path(args.output))
    else:
        output_path = path.with_name(f"{path.stem}_plot.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_counts(summary, path.name, output_path)

    print(f"\nPlot saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
