import time
import random
import requests
import csv
import os


# =========================
# EXPERIMENT SETTINGS
# =========================

WORD_LENGTH           = 5
WORDS_PER_RUN         = 20
NUMBER_OF_RUNS        = 3

SECONDS_PER_WORD      = 0.5

PAUSE_BEFORE_RECALL   = False
RECALL_DELAY_SECONDS  = 30

MINIMUM_FREQUENCY     = 3

CSV_FILE              = "memory_results.csv"


# =========================
# MENU
# =========================

def select_mode():
    while True:
        print("\nSelect experiment mode:")
        print("1. Free Recall")
        print("2. Serial Recall")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            return "free_recall"

        if choice == "2":
            return "serial_recall"

        if choice == "3":
            return None

        print("Invalid choice.")


# =========================
# WORD GENERATION
# =========================

def get_available_words():
    pattern = "?" * WORD_LENGTH
    url = f"https://api.datamuse.com/words?sp={pattern}&md=f&max=1000"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    available_words = []

    for item in data:
        word = item["word"].strip().lower()

        if len(word) != WORD_LENGTH or not word.isalpha():
            continue

        frequency = 0

        for tag in item.get("tags", []):
            if tag.startswith("f:"):
                frequency = float(tag[2:])
                break

        if frequency >= MINIMUM_FREQUENCY:
            available_words.append(word)

    return list(set(available_words))


# =========================
# RUN NUMBER
# =========================

def get_next_run_number(mode):
    if not os.path.exists(CSV_FILE):
        return 1

    highest_run = 0

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                if row["mode"] != mode:
                    continue

                run_number = int(row["run"])
                highest_run = max(highest_run, run_number)

            except (ValueError, KeyError):
                continue

    return highest_run + 1


# =========================
# WORD DISPLAY
# =========================

def display_words(words):
    for word in words:
        print(f"{word:<30}", end="\r", flush=True)
        time.sleep(SECONDS_PER_WORD)

    print(" " * 30, end="\r")

    if PAUSE_BEFORE_RECALL:
        time.sleep(RECALL_DELAY_SECONDS)


# =========================
# FREE RECALL
# =========================

def free_recall(words):
    remembered_words = []

    print("\nEnter the words you remember in any order.")
    print("Type 'exit' when finished.\n")

    while True:
        remembered_word = input("Word: ").strip().lower()

        if remembered_word == "exit":
            break

        if remembered_word:
            remembered_words.append(remembered_word)

    results = []

    for position, word in enumerate(words, start=1):
        remembered = word in remembered_words

        results.append({
            "position":       position,
            "word":           word,
            "recalled_word":  "",
            "remembered":     remembered
        })

    return results


# =========================
# SERIAL RECALL
# =========================

def serial_recall(words):
    recalled_words = []

    print("\nEnter the words in the order they were shown.")
    print("Press Enter to leave a position blank.\n")

    for position in range(1, len(words) + 1):
        recalled_word = input(
            f"Position {position:>2}: "
        ).strip().lower()

        recalled_words.append(recalled_word)

    results = []

    for position, word in enumerate(words, start=1):
        recalled_word = recalled_words[position - 1]
        remembered = recalled_word == word

        results.append({
            "position":       position,
            "word":           word,
            "recalled_word":  recalled_word,
            "remembered":     remembered
        })

    return results


# =========================
# RESULTS
# =========================

def print_results(mode, run_number, results):
    print(f"\nResults for Run {run_number}")
    print("-" * 50)

    if mode == "free_recall":
        for result in results:
            status = (
                "Remembered"
                if result["remembered"]
                else "Not remembered"
            )

            print(
                f"{result['position']:>2}. "
                f"{result['word']:<10} "
                f"{status}"
            )

    elif mode == "serial_recall":
        for result in results:
            status = (
                "Correct"
                if result["remembered"]
                else "Incorrect"
            )

            recalled_word = result["recalled_word"]

            if not recalled_word:
                recalled_word = "-"

            print(
                f"{result['position']:>2}. "
                f"Expected: {result['word']:<10} "
                f"Recalled: {recalled_word:<10} "
                f"{status}"
            )

    remembered_count = sum(
        result["remembered"] for result in results
    )

    print(
        f"\nScore: {remembered_count}/{len(results)}"
    )


# =========================
# CSV
# =========================

def save_results(mode, run_number, results):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "run",
                "position",
                "word",
                "recalled_word",
                "remembered"
            ]
        )

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow({
                "mode":           mode,
                "run":            run_number,
                "position":       result["position"],
                "word":           result["word"],
                "recalled_word":  result["recalled_word"],
                "remembered":     result["remembered"]
            })


# =========================
# MAIN PROGRAM
# =========================

def main():
    available_words = get_available_words()

    while True:
        mode = select_mode()

        if mode is None:
            break

        start_run = get_next_run_number(mode)

        for run_offset in range(NUMBER_OF_RUNS):
            run_number = start_run + run_offset

            print(f"\nRun {run_number}")
            print("-" * 30)

            words = random.sample(
                available_words,
                WORDS_PER_RUN
            )

            display_words(words)

            if mode == "free_recall":
                results = free_recall(words)

            else:
                results = serial_recall(words)

            print_results(
                mode,
                run_number,
                results
            )

            save_results(
                mode,
                run_number,
                results
            )

            print(
                f"\nRun {run_number} saved to {CSV_FILE}."
            )


if __name__ == "__main__":
    main()