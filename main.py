import time
import random
import requests
import csv
import os


# =========================
# EXPERIMENT SETTINGS
# =========================

WORD_LENGTH                 = 5
WORDS_PER_RUN               = 20
NUMBER_OF_RUNS              = 5

SLOW_SECONDS_PER_WORD       = 3.0
FAST_SECONDS_PER_WORD       = 1.0

PAUSE_DURATION_SECONDS      = 30
MATH_TASK_DURATION_SECONDS  = 30

MATH_MINIMUM_NUMBER         = 2
MATH_MAXIMUM_NUMBER         = 19

MINIMUM_FREQUENCY           = 1

CSV_FILE                    = "memory_results.csv"


# =========================
# EXPERIMENT MODES
# =========================

MODES = {
    "1": {
        "name": "free_recall_slow",
        "recall_type": "free_recall",
        "seconds_per_word": SLOW_SECONDS_PER_WORD,
        "post_sequence_task": "none"
    },
    "2": {
        "name": "free_recall_fast",
        "recall_type": "free_recall",
        "seconds_per_word": FAST_SECONDS_PER_WORD,
        "post_sequence_task": "none"
    },
    "3": {
        "name": "free_recall_working_memory",
        "recall_type": "free_recall",
        "seconds_per_word": SLOW_SECONDS_PER_WORD,
        "post_sequence_task": "multiplication"
    },
    "4": {
        "name": "free_recall_pause",
        "recall_type": "free_recall",
        "seconds_per_word": SLOW_SECONDS_PER_WORD,
        "post_sequence_task": "pause"
    },
    "5": {
        "name": "serial_recall",
        "recall_type": "serial_recall",
        "seconds_per_word": SLOW_SECONDS_PER_WORD,
        "post_sequence_task": "none"
    }
}


# =========================
# MENU
# =========================

def select_mode():
    while True:
        print("\nMemory Experiment")
        print("=" * 40)

        print("1. Free Recall - Slow")
        print("2. Free Recall - Fast")
        print("3. Free Recall - Working Memory Task")
        print("4. Free Recall - Pause")
        print("5. Serial Recall")
        print("6. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "6":
            return None

        if choice in MODES:
            return MODES[choice]

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

    response = requests.get(
        url,
        timeout=10
    )

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

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            try:
                run_number = int(row["run"])

                highest_run = max(
                    highest_run,
                    run_number
                )

            except (ValueError, KeyError):
                continue

    return highest_run + 1


# =========================
# WORD SELECTION
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


# =========================
# DISPLAY
# =========================

def clear_line():
    print(
        "\r" + " " * 60 + "\r",
        end="",
        flush=True
    )


def wait_for_run_start(run_number, first_run):
    if first_run:
        input(
            f"\nPress Enter to start Run {run_number}..."
        )

    else:
        input(
            f"\nPress Enter to start the next run "
            f"(Run {run_number})..."
        )


def present_words(words, seconds_per_word):
    clear_line()

    for word in words:
        print(
            f"\r{word:<30}",
            end="",
            flush=True
        )

        time.sleep(seconds_per_word)

    clear_line()


# =========================
# PAUSE CONDITION
# =========================

def run_pause():
    print(
        f"\nWait {PAUSE_DURATION_SECONDS} seconds "
        f"before recalling the words."
    )

    time.sleep(PAUSE_DURATION_SECONDS)

    clear_line()


# =========================
# WORKING MEMORY CONDITION
# =========================

def run_multiplication_task():
    print("\nAnswer the multiplication questions.")
    print(
        f"Continue for approximately "
        f"{MATH_TASK_DURATION_SECONDS} seconds.\n"
    )

    correct_answers = 0
    total_questions = 0

    start_time = time.perf_counter()

    while (
        time.perf_counter() - start_time
        < MATH_TASK_DURATION_SECONDS
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
# POST-SEQUENCE TASK
# =========================

def run_post_sequence_task(task):
    if task == "none":
        return 0, 0

    if task == "pause":
        run_pause()

        return 0, 0

    if task == "multiplication":
        return run_multiplication_task()

    return 0, 0


# =========================
# FREE RECALL
# =========================

def collect_free_recall():
    recalled_words = []

    print("\nEnter every word you remember.")
    print("Order does not matter.")
    print("Type 'exit' when finished.\n")

    while True:
        recalled_word = input(
            "Word: "
        ).strip().lower()

        if recalled_word == "exit":
            break

        if recalled_word:
            recalled_words.append(
                recalled_word
            )

    return recalled_words


def score_free_recall(words, recalled_words):
    results = []

    for position, word in enumerate(
        words,
        start=1
    ):
        remembered = word in recalled_words

        recall_position = ""

        if remembered:
            recall_position = (
                recalled_words.index(word) + 1
            )

        results.append({
            "position": position,
            "word": word,
            "recalled_word": "",
            "recall_position": recall_position,
            "remembered": remembered
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
        "a position.\n"
    )

    for position in range(
        1,
        number_of_words + 1
    ):
        recalled_word = input(
            f"Position {position:>2}: "
        ).strip().lower()

        recalled_words.append(
            recalled_word
        )

    return recalled_words


def score_serial_recall(
    words,
    recalled_words
):
    results = []

    for position, word in enumerate(
        words,
        start=1
    ):
        recalled_word = (
            recalled_words[position - 1]
        )

        remembered = (
            recalled_word == word
        )

        results.append({
            "position": position,
            "word": word,
            "recalled_word": recalled_word,
            "recall_position": position,
            "remembered": remembered
        })

    return results


# =========================
# PRINT RESULTS
# =========================

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
        if result["recalled_word"]:
            recalled_word = (
                result["recalled_word"]
            )

        else:
            recalled_word = "-"

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


def print_results(
    mode,
    run_number,
    results,
    math_correct,
    math_total
):
    print(f"\nResults for Run {run_number}")
    print("-" * 60)

    if mode["recall_type"] == "free_recall":
        print_free_recall_results(
            results
        )

    else:
        print_serial_recall_results(
            results
        )

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


# =========================
# CSV
# =========================

def get_post_sequence_duration(task):
    if task == "pause":
        return PAUSE_DURATION_SECONDS

    if task == "multiplication":
        return MATH_TASK_DURATION_SECONDS

    return 0


def save_results(
    run_number,
    mode,
    results,
    math_correct,
    math_total
):
    file_exists = os.path.exists(
        CSV_FILE
    )

    fieldnames = [
        "run",
        "mode",
        "recall_type",
        "seconds_per_word",
        "post_sequence_task",
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

                "mode":
                    mode["name"],

                "recall_type":
                    mode["recall_type"],

                "seconds_per_word":
                    mode["seconds_per_word"],

                "post_sequence_task":
                    mode["post_sequence_task"],

                "post_sequence_duration":
                    get_post_sequence_duration(
                        mode["post_sequence_task"]
                    ),

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

        # Force the results to be written to disk immediately
        file.flush()
        os.fsync(file.fileno())


# =========================
# SINGLE RUN
# =========================

def run_experiment(
    run_number,
    mode,
    available_words
):
    print(f"\nRun {run_number}")
    print(f"Mode: {mode['name']}")
    print("-" * 40)

    words = select_words(
        available_words
    )

    present_words(
        words,
        mode["seconds_per_word"]
    )

    math_correct, math_total = (
        run_post_sequence_task(
            mode["post_sequence_task"]
        )
    )

    if mode["recall_type"] == "free_recall":
        recalled_words = (
            collect_free_recall()
        )

        results = score_free_recall(
            words,
            recalled_words
        )

    else:
        recalled_words = (
            collect_serial_recall(
                len(words)
            )
        )

        results = score_serial_recall(
            words,
            recalled_words
        )

    print_results(
        mode,
        run_number,
        results,
        math_correct,
        math_total
    )

    # Save immediately after this run is completed
    save_results(
        run_number,
        mode,
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
    available_words = (
        get_available_words()
    )

    print(
        f"Available word pool: "
        f"{len(available_words)} words"
    )

    while True:
        mode = select_mode()

        if mode is None:
            break

        start_run = (
            get_next_run_number()
        )

        for run_offset in range(
            NUMBER_OF_RUNS
        ):
            run_number = (
                start_run + run_offset
            )

            wait_for_run_start(
                run_number,
                run_offset == 0
            )

            run_experiment(
                run_number,
                mode,
                available_words
            )


if __name__ == "__main__":
    main()
