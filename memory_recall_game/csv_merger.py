import argparse
import csv
from pathlib import Path


def detect_delimiter(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def clean_path(value):
    return value.strip().strip('"').strip("'")


def collect_files():
    print("CSV Merger")
    print("Enter one CSV path per line. Press Enter on an empty line when finished.\n")

    paths = []
    while True:
        value = input(f"CSV file {len(paths) + 1}: ").strip()
        if not value:
            break
        paths.append(clean_path(value))

    return paths


def merge_csv_files(file_paths, output_path):
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
                raise ValueError(f"Columns do not match in: {file_path}")

            all_rows.extend(reader)

    if not all_rows or header is None:
        raise ValueError("No data was found in the supplied CSV files.")

    possible_columns = {
        "run",
        "run_number",
        "run number",
        "trial",
        "trial_number",
        "trial number",
        "entry",
        "entry_number",
        "entry number",
    }

    number_column = None
    for column in header:
        if column.strip().lower() in possible_columns:
            number_column = column
            break

    if number_column is None:
        raise ValueError("Could not find a run/trial/entry number column.")

    for i, row in enumerate(all_rows, start=1):
        row[number_column] = str(i)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows), number_column


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV files and renumber the run/trial/entry column continuously."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="CSV files to merge. If omitted, the program asks for paths interactively."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Default: memory_results_merged.csv next to the first input file."
    )
    args = parser.parse_args()

    file_paths = [clean_path(p) for p in args.csv_files]
    if not file_paths:
        file_paths = collect_files()

    if not file_paths:
        print("No CSV files supplied.")
        return 1

    missing = [p for p in file_paths if not Path(p).is_file()]
    if missing:
        print("\nThe following file(s) were not found:")
        for path in missing:
            print(f"  {path}")
        return 1

    if args.output:
        output_path = Path(clean_path(args.output))
    else:
        default_output = Path(file_paths[0]).resolve().parent / "memory_results_merged.csv"
        entered = input(f"Output file [{default_output}]: ").strip()
        output_path = Path(clean_path(entered)) if entered else default_output

    try:
        count, number_column = merge_csv_files(file_paths, output_path)
    except Exception as exc:
        print(f"\nMerge failed: {exc}")
        return 1

    print("\nFinished.")
    print(f"Merged entries: {count}")
    print(f"Renumbered column: {number_column}")
    print(f"Numbers: 1-{count}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
