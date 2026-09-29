import csv
import tkinter as tk
from tkinter import filedialog, messagebox


def detect_delimiter(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def main():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Vælg CSV-filer",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_paths:
        return

    all_rows = []
    header = None
    delimiter = None

    for file_path in file_paths:
        current_delimiter = detect_delimiter(file_path)

        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=current_delimiter)

            if reader.fieldnames is None:
                continue

            if header is None:
                header = reader.fieldnames
                delimiter = current_delimiter

            elif reader.fieldnames != header:
                messagebox.showerror(
                    "Fejl",
                    f"Kolonnerne matcher ikke i:\n{file_path}"
                )
                return

            all_rows.extend(reader)

    if not all_rows:
        messagebox.showerror("Fejl", "Der blev ikke fundet nogen data.")
        return

    possible_columns = [
        "run",
        "run_number",
        "run number",
        "trial",
        "trial_number",
        "trial number",
        "entry",
        "entry_number",
        "entry number"
    ]

    number_column = None

    for column in header:
        if column.strip().lower() in possible_columns:
            number_column = column
            break

    if number_column is None:
        messagebox.showerror(
            "Fejl",
            "Kunne ikke finde en run/trial-kolonne."
        )
        return

    for i, row in enumerate(all_rows, start=1):
        row[number_column] = str(i)

    output_path = filedialog.asksaveasfilename(
        title="Gem samlet CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="memory_results_merged.csv"
    )

    if not output_path:
        return

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=header,
            delimiter=delimiter
        )

        writer.writeheader()
        writer.writerows(all_rows)

    messagebox.showinfo(
        "Færdig",
        f"{len(all_rows)} entries blev samlet og renummereret.\n\n"
        f"Kolonne: {number_column}\n"
        f"Numre: 1-{len(all_rows)}"
    )


if __name__ == "__main__":
    main()
