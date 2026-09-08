import time
import random
import requests
import csv
import os


# =========================
# EXPERIMENT SETTINGS
# =========================

WORD_LENGTH                  = 5
WORDS_PER_RUN                = 20
NUMBER_OF_RUNS               = 3

SECONDS_PER_WORD             = 1.0
MINIMUM_FREQUENCY            = 1

POST_SEQUENCE_DURATION       = 30
MATH_MINIMUM_NUMBER          = 2
MATH_MAXIMUM_NUMBER          = 19

CSV_FILE                     = "memory_results.csv"


# =========================
# MENUS
# =========================

def select_recall_mode():
    while True:
        print("\nRecall mode")
        print("1. Free recall")
        print("2. Serial recall")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            return "free_recall"

        if choice == "2":
            return "serial_recall"

        if choice == "3":
            return None

        print("Invalid choice.")


def select_post_sequence_condition():
    while True:
        print("\nPost-sequence condition")
        print("1. Immediate recall")
        print("2. Pause")
        print("3. Multiplication task")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            return "immediate"

        if choice == "2":
            return "pause"

        if choice == "3":
            return "multiplication"

        print("Invalid choice.")


# =========================
# WORD POOL
# =========================

def get_available_words():
    pattern = "?" * WORD_LENGTH
    url = (
        f"https://api.datamuse.com/words"
        f"?sp={pattern}&md=f&max=1000"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    available_words = []

    for item in data:
        word = item["word"].strip().lower()

        if len(word) != WORD_LENGTH:
            continue

        if not word.isalpha():
            continue

        frequency = 0

        for tag in item.get("tags", []):
            if tag.startswith("f:"):
                frequency = float(tag[2:])
                break

        if frequency >= MINIMUM_FREQUENCY:
            available_words.append(word)

    return sorted(set(available_words))


# =========================
# RUN NUMBER
# =========================

def get_next_run_number():
    if not os.path.exists(CSV_FILE):
        return 1

    highest_run = 0

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                run_number = int(row["run"])
                highest_run = max(highest_run, run_number)

            except (ValueError, KeyError):
                continue

    return highest_run + 1


# =========================
# WORD PRESENTATION
# =========================

def select_words(available_words):
    if len(available_words) < WORDS_PER_RUN:
        raise ValueError(
            f"Only {len(available_words)} words are available, "
            f"but {WORDS_PER_RUN} are required."
        )

    return random.sample(
        available_words,
        WORDS_PER_RUN
    )


def present_words(words):
    print("\nGet ready...")
    time.sleep(1)

    for word in words:
        print(
            f"\r{word:<30}",
            end="",
            flush=True
        )

        time.sleep(SECONDS_PER_WORD)

    clear_line()


def clear_line():
    print(
        "\r" + " " * 50 + "\r",
        end="",
        flush=True
    )


# =========================
# POST-SEQUENCE CONDITIONS
# =========================

def run_post_sequence_condition(condition):
    if condition == "immediate":
        return 0, 0

    if condition == "pause":
        run_pause()
        return 0, 0

    if condition == "multiplication":
        return run_multiplication_task()

    return 0, 0


def run_pause():
    print(
        f"Please wait {POST_SEQUENCE_DURATION} seconds..."
    )

    time.sleep(POST_SEQUENCE_DURATION)

    clear_line()


def run_multiplication_task():
    print("\nAnswer the multiplication questions.")
    print(
        f"The task lasts approximately "
        f"{POST_SEQUENCE_DURATION} seconds.\n"
    )

    correct_answers = 0
    total_questions = 0

    start_time = time.perf_counter()

    while (
        time.perf_counter() - start_time
        < POST_SEQUENCE_DURATION
    ):
        first_number = random.randint(
            MATH_MINIMUM_NUMBER,
            MATH_MAXIMUM_NUMBER
        )

        second_number = random.randint(
            MATH_MINIMUM_NUMBER,
            MATH_MAXIMUM_NUMBER
        )

        correct_answer = (
            first_number * second_number
        )

        answer = input(
            f"{first_number} x {second_number} = "
        ).strip()

        total_questions += 1

        try:
            if int(answer) == correct_answer:
                correct_answers += 1

        except ValueError:
            pass

    print("\nTime is up.")

    return correct_answers, total_questions


# =========================
# FREE RECALL
# =========================

def collect_free_recall():
    recalled_words = []

    print("\nEnter every word you remember.")
    print("The order does not matter.")
    print("Type 'exit' when you are finished.\n")

    while True:
        recalled_word = input("Word: ").strip().lower()

        if recalled_word == "exit":
            break

        if recalled_word:
            recalled_words.append(recalled_word)

    return recalled_words


def score_free_recall(words, recalled_words):
    results = []

    for position, word in enumerate(words, start=1):
        remembered = word in recalled_words

        recall_position = ""

        if remembered:
            recall_position = (
                recalled_words.index(word) + 1
            )

        results.append({
            "position":        position,
            "word":            word,
            "recalled_word":   "",
            "recall_position": recall_position,
            "remembered":      remembered
        })

    return results


# =========================
# SERIAL RECALL
# =========================

def collect_serial_recall(number_of_words):
    recalled_words = []

    print("\nEnter the words in the order shown.")
    print(
        "Press Enter if you cannot remember "
        "a particular position.\n"
    )

    for position in range(1, number_of_words + 1):
        recalled_word = input(
            f"Position {position:>2}: "
        ).strip().lower()

        recalled_words.append(recalled_word)

    return recalled_words


def score_serial_recall(words, recalled_words):
    results = []

    for position, word in enumerate(words, start=1):
        recalled_word = recalled_words[position - 1]

        remembered = (
            recalled_word == word
        )

        results.append({
            "position":        position,
            "word":            word,
            "recalled_word":   recalled_word,
            "recall_position": position,
            "remembered":      remembered
        })

    return results


# =========================
# RESULTS
# =========================

def print_results(
    recall_mode,
    run_number,
    results,
    math_correct,
    math_total
):
    print(f"\nResults for run {run_number}")
    print("-" * 60)

    if recall_mode == "free_recall":
        print_free_recall_results(results)

    elif recall_mode == "serial_recall":
        print_serial_recall_results(results)

    remembered_count = sum(
        result["remembered"]
        for result in results
    )

    print(
        f"\nMemory score: "
        f"{remembered_count}/{len(results)}"
    )

    if math_total > 0:
        print(
            f"Math score: "
            f"{math_correct}/{math_total}"
        )


def print_free_recall_results(results):
    for result in results:
        if result["remembered"]:
            status = "Remembered"
        else:
            status = "Not remembered"

        print(
            f"{result['position']:>2}. "
            f"{result['word']:<10} "
            f"{status}"
        )


def print_serial_recall_results(results):
    for result in results:
        recalled_word = (
            result["recalled_word"]
            if result["recalled_word"]
            else "-"
        )

        if result["remembered"]:
            status = "Correct"
        else:
            status = "Incorrect"

        print(
            f"{result['position']:>2}. "
            f"Expected: {result['word']:<10} "
            f"Recalled: {recalled_word:<10} "
            f"{status}"
        )


# =========================
# CSV
# =========================

def save_results(
    run_number,
    recall_mode,
    condition,
    results,
    math_correct,
    math_total
):
    file_exists = os.path.exists(CSV_FILE)

    fieldnames = [
        "run",
        "recall_mode",
        "condition",
        "seconds_per_word",
        "post_sequence_duration",
        "position",
        "word",
        "recalled_word",
        "recall_position",
        "remembered",
        "math_correct",
        "math_total"
    ]

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow({
                "run":
                    run_number,

                "recall_mode":
                    recall_mode,

                "condition":
                    condition,

                "seconds_per_word":
                    SECONDS_PER_WORD,

                "post_sequence_duration":
                    get_condition_duration(condition),

                "position":
                    result["position"],

                "word":
                    result["word"],

                "recalled_word":
                    result["recalled_word"],

                "recall_position":
                    result["recall_position"],

                "remembered":
                    result["remembered"],

                "math_correct":
                    math_correct,

                "math_total":
                    math_total
            })


def get_condition_duration(condition):
    if condition in ["pause", "multiplication"]:
        return POST_SEQUENCE_DURATION

    return 0


# =========================
# SINGLE RUN
# =========================

def run_experiment(
    run_number,
    recall_mode,
    condition,
    available_words
):
    print(f"\nRun {run_number}")
    print("-" * 30)

    words = select_words(available_words)

    present_words(words)

    math_correct, math_total = (
        run_post_sequence_condition(condition)
    )

    if recall_mode == "free_recall":
        recalled_words = collect_free_recall()

        results = score_free_recall(
            words,
            recalled_words
        )

    else:
        recalled_words = collect_serial_recall(
            len(words)
        )

        results = score_serial_recall(
            words,
            recalled_words
        )

    print_results(
        recall_mode,
        run_number,
        results,
        math_correct,
        math_total
    )

    save_results(
        run_number,
        recall_mode,
        condition,
        results,
        math_correct,
        math_total
    )

    print(
        f"\nRun {run_number} saved to {CSV_FILE}."
    )


# =========================
# MAIN
# =========================

def main():
    available_words = get_available_words()

    print(
        f"Available word pool: "
        f"{len(available_words)} words"
    )

    while True:
        recall_mode = select_recall_mode()

        if recall_mode is None:
            break

        condition = select_post_sequence_condition()

        start_run = get_next_run_number()

        for run_offset in range(NUMBER_OF_RUNS):
            run_number = (
                start_run + run_offset
            )

            run_experiment(
                run_number,
                recall_mode,
                condition,
                available_words
            )


if __name__ == "__main__":
    main()
