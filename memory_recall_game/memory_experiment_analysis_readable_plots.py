import sys
import math
import csv
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}

def ensure_packages():
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )

ensure_packages()

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHA = 0.05
CONFIDENCE = 0.95
FINGER_TAPPING_EQUIVALENCE_MARGIN = 1.0

MODE_LABELS = {
    "free_recall_slow": "Free recall: slow",
    "free_recall_fast": "Free recall: fast",
    "free_recall_working_memory": "Free recall: working memory task",
    "free_recall_pause": "Free recall: pause",
    "serial_recall": "Serial recall: baseline",
    "serial_recall_chunking": "Serial recall: chunking",
    "serial_recall_articulatory_suppresion": "Serial recall: articulatory suppression",
    "serial_recall_articulatory_suppression": "Serial recall: articulatory suppression",
    "serial_recall_finger_tapping": "Serial recall: finger tapping"
}


def find_participant_column(df):
    candidates = [
        "participant",
        "participant_id",
        "subject",
        "subject_id",
        "person",
        "name",
        "source_file"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def clean_bool(series):
    if series.dtype == bool:
        return series

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False
    }
    converted = text.map(mapping)
    if converted.notna().all():
        return converted.astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(bool)

    raise ValueError(
        "Could not interpret the 'remembered' column as True/False values."
    )


def detect_csv_delimiter(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        counts = {delimiter: first_line.count(delimiter) for delimiter in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def load_csv_files(paths):
    frames = []

    for path in paths:
        delimiter = detect_csv_delimiter(path)
        frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")

        participant_col = find_participant_column(frame)

        if len(paths) > 1 and participant_col is None:
            frame["participant"] = Path(path).stem
        elif len(paths) > 1 and participant_col is not None:
            frame["participant"] = frame[participant_col].astype(str)
        elif participant_col is not None and participant_col != "participant":
            frame["participant"] = frame[participant_col].astype(str)

        frame["_source_file"] = Path(path).name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    required = {
        "mode",
        "recall_type",
        "position",
        "remembered",
        "word",
        "recalled_word"
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["remembered"] = clean_bool(df["remembered"])

    if df["position"].isna().any():
        raise ValueError("Some values in 'position' are not numeric.")

    return df


def add_trial_ids(df):
    df = df.copy()
    participant_col = "participant" if "participant" in df.columns else None

    if participant_col:
        trial_ids = []
        for participant, group in df.groupby(participant_col, sort=False):
            counter = group["position"].eq(1).cumsum()
            trial_ids.append(pd.Series(counter.values, index=group.index))
        df["_trial_id"] = pd.concat(trial_ids).sort_index().astype(int)
    else:
        df["_trial_id"] = df["position"].eq(1).cumsum().astype(int)

    return df


def t_mean_ci(values, confidence=CONFIDENCE):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, np.nan, np.nan

    se = stats.sem(values)
    critical = stats.t.ppf(
        (1.0 + confidence) / 2.0,
        df=len(values) - 1
    )
    half = critical * se

    return mean, mean - half, mean + half


def wilson_ci(successes, total, confidence=CONFIDENCE):
    if total <= 0:
        return np.nan, np.nan

    z = stats.norm.ppf((1.0 + confidence) / 2.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denom
    )
    return center - half, center + half


def paired_difference_test(a, b, alternative="two-sided"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    diff = a - b

    if len(diff) < 2:
        return None

    mean_diff = float(np.mean(diff))
    se = stats.sem(diff)
    dfree = len(diff) - 1
    statistic = mean_diff / se if se > 0 else np.inf

    if alternative == "greater":
        p_value = stats.t.sf(statistic, dfree)
    elif alternative == "less":
        p_value = stats.t.cdf(statistic, dfree)
    else:
        p_value = 2.0 * stats.t.sf(abs(statistic), dfree)

    critical = stats.t.ppf(0.975, dfree)
    ci_low = mean_diff - critical * se
    ci_high = mean_diff + critical * se

    return {
        "estimate": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "statistic": statistic,
        "df": dfree,
        "p_value": p_value,
        "n": len(diff),
        "test": "Paired t-test"
    }


def welch_difference_test(a, b, alternative="two-sided"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return None

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    se2 = var_a / len(a) + var_b / len(b)
    se = math.sqrt(se2)

    numerator = se2 * se2
    denominator = (
        (var_a / len(a)) ** 2 / (len(a) - 1)
        + (var_b / len(b)) ** 2 / (len(b) - 1)
    )
    dfree = numerator / denominator

    mean_diff = mean_a - mean_b
    statistic = mean_diff / se if se > 0 else np.inf

    if alternative == "greater":
        p_value = stats.t.sf(statistic, dfree)
    elif alternative == "less":
        p_value = stats.t.cdf(statistic, dfree)
    else:
        p_value = 2.0 * stats.t.sf(abs(statistic), dfree)

    critical = stats.t.ppf(0.975, dfree)
    ci_low = mean_diff - critical * se
    ci_high = mean_diff + critical * se

    return {
        "estimate": float(mean_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "statistic": float(statistic),
        "df": float(dfree),
        "p_value": float(p_value),
        "n": f"{len(a)} + {len(b)}",
        "test": "Welch independent-samples t-test"
    }


def one_sample_test(values, null_mean, alternative="two-sided"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return None

    mean = np.mean(values)
    se = stats.sem(values)
    dfree = len(values) - 1
    statistic = (mean - null_mean) / se if se > 0 else np.inf

    if alternative == "less":
        p_value = stats.t.cdf(statistic, dfree)
    elif alternative == "greater":
        p_value = stats.t.sf(statistic, dfree)
    else:
        p_value = 2.0 * stats.t.sf(abs(statistic), dfree)

    critical = stats.t.ppf(0.975, dfree)
    ci_low = mean - critical * se
    ci_high = mean + critical * se

    return {
        "estimate": float(mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "statistic": float(statistic),
        "df": float(dfree),
        "p_value": float(p_value),
        "n": len(values),
        "test": "One-sample t-test"
    }


def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def decision_from_p(p):
    if pd.isna(p):
        return "Not enough data"
    if p < ALPHA:
        return "Reject H0: statistically significant evidence against the no-effect/no-difference hypothesis"
    return "Do not reject H0: not enough evidence to conclude there is an effect/difference"


def trial_scores(df):
    group_cols = []
    if "participant" in df.columns:
        group_cols.append("participant")
    group_cols.extend(["_trial_id", "mode"])

    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            score=("remembered", "sum"),
            list_length=("remembered", "size")
        )
    )


def analysis_units_for_mode(score_df, mode):
    subset = score_df[score_df["mode"] == mode].copy()

    if "participant" in score_df.columns:
        return (
            subset.groupby("participant", as_index=False)["score"]
            .mean()
            .rename(columns={"score": "value"})
        )

    subset = subset[["_trial_id", "score"]].copy()
    subset = subset.rename(columns={"_trial_id": "unit", "score": "value"})
    return subset


def compare_modes(score_df, mode_a, mode_b):
    if "participant" in score_df.columns:
        pivot = score_df.pivot_table(
            index="participant",
            columns="mode",
            values="score",
            aggfunc="mean"
        )
        if mode_a not in pivot.columns or mode_b not in pivot.columns:
            return None

        pairs = pivot[[mode_a, mode_b]].dropna()
        return paired_difference_test(
            pairs[mode_a].values,
            pairs[mode_b].values,
            alternative="two-sided"
        )

    a = score_df.loc[score_df["mode"] == mode_a, "score"].values
    b = score_df.loc[score_df["mode"] == mode_b, "score"].values
    return welch_difference_test(a, b, alternative="two-sided")


def region_effect_test(df, region_name):
    baseline = df[df["mode"] == "free_recall_slow"].copy()
    if baseline.empty:
        return None

    trial_group_cols = []
    if "participant" in baseline.columns:
        trial_group_cols.append("participant")
    trial_group_cols.append("_trial_id")

    rows = []

    for keys, group in baseline.groupby(trial_group_cols, sort=False):
        first = group.loc[group["position"].between(1, 3), "remembered"].mean()
        middle = group.loc[group["position"].between(6, 10), "remembered"].mean()
        last = group.loc[group["position"].between(13, 15), "remembered"].mean()

        row = {
            "first": first,
            "middle": middle,
            "last": last
        }

        if "participant" in baseline.columns:
            if isinstance(keys, tuple):
                row["participant"] = keys[0]
            else:
                row["participant"] = keys

        rows.append(row)

    region_df = pd.DataFrame(rows)

    if "participant" in baseline.columns:
        region_df = (
            region_df.groupby("participant", as_index=False)[["first", "middle", "last"]]
            .mean()
        )

    target = "first" if region_name == "primacy" else "last"

    return paired_difference_test(
        region_df[target].values,
        region_df["middle"].values,
        alternative="greater"
    )


def serial_capacity_test(score_df):
    baseline = score_df[score_df["mode"] == "serial_recall"].copy()
    if baseline.empty:
        return None

    if "participant" in score_df.columns:
        values = baseline.groupby("participant")["score"].mean().values
    else:
        values = baseline["score"].values

    list_length = int(round(baseline["list_length"].median()))
    result = one_sample_test(values, list_length, alternative="less")

    if result:
        result["null_mean"] = list_length

    return result


def classify_serial_errors(df):
    serial = df[df["recall_type"] == "serial_recall"].copy()

    if serial.empty:
        return pd.DataFrame(columns=["error_type", "count", "proportion", "ci_low", "ci_high"])

    group_cols = []
    if "participant" in serial.columns:
        group_cols.append("participant")
    group_cols.append("_trial_id")

    counts = {
        "Omission": 0,
        "Transposition": 0,
        "Intrusion/substitution": 0
    }

    for _, group in serial.groupby(group_cols, sort=False):
        target_words = set(
            group["word"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        for _, row in group.iterrows():
            if bool(row["remembered"]):
                continue

            response = row["recalled_word"]

            if pd.isna(response) or str(response).strip() == "":
                counts["Omission"] += 1
                continue

            response_text = str(response).strip().lower()
            target_text = str(row["word"]).strip().lower()

            if response_text != target_text and response_text in target_words:
                counts["Transposition"] += 1
            else:
                counts["Intrusion/substitution"] += 1

    total = sum(counts.values())
    rows = []

    for name, count in counts.items():
        if total > 0:
            low, high = wilson_ci(count, total)
            proportion = count / total
        else:
            low, high, proportion = np.nan, np.nan, np.nan

        rows.append(
            {
                "error_type": name,
                "count": count,
                "proportion": proportion,
                "ci_low": low,
                "ci_high": high
            }
        )

    return pd.DataFrame(rows)


def error_type_chi_square(error_df):
    counts = error_df["count"].to_numpy(dtype=float)

    if counts.sum() <= 0 or len(counts) < 2:
        return None

    statistic, p_value = stats.chisquare(counts)

    return {
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "statistic": float(statistic),
        "df": len(counts) - 1,
        "p_value": float(p_value),
        "n": int(counts.sum()),
        "test": "Chi-square goodness-of-fit test"
    }


def equivalence_tost(score_df, mode_a, mode_b, margin):
    if "participant" in score_df.columns:
        pivot = score_df.pivot_table(
            index="participant",
            columns="mode",
            values="score",
            aggfunc="mean"
        )

        if mode_a not in pivot.columns or mode_b not in pivot.columns:
            return None

        pairs = pivot[[mode_a, mode_b]].dropna()
        diff = (pairs[mode_a] - pairs[mode_b]).to_numpy(dtype=float)

        if len(diff) < 2:
            return None

        estimate = np.mean(diff)
        se = stats.sem(diff)
        dfree = len(diff) - 1

        lower_t = (estimate + margin) / se
        upper_t = (estimate - margin) / se
        p_lower = stats.t.sf(lower_t, dfree)
        p_upper = stats.t.cdf(upper_t, dfree)
        p_tost = max(p_lower, p_upper)

        critical_95 = stats.t.ppf(0.975, dfree)
        critical_90 = stats.t.ppf(0.95, dfree)

        ci95 = (
            estimate - critical_95 * se,
            estimate + critical_95 * se
        )
        ci90 = (
            estimate - critical_90 * se,
            estimate + critical_90 * se
        )

        return {
            "estimate": float(estimate),
            "ci_low": float(ci95[0]),
            "ci_high": float(ci95[1]),
            "ci90_low": float(ci90[0]),
            "ci90_high": float(ci90[1]),
            "statistic": np.nan,
            "df": float(dfree),
            "p_value": float(p_tost),
            "p_lower": float(p_lower),
            "p_upper": float(p_upper),
            "n": len(diff),
            "test": "TOST paired equivalence test"
        }

    a = score_df.loc[score_df["mode"] == mode_a, "score"].to_numpy(dtype=float)
    b = score_df.loc[score_df["mode"] == mode_b, "score"].to_numpy(dtype=float)

    if len(a) < 2 or len(b) < 2:
        return None

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    estimate = mean_a - mean_b

    se2 = var_a / len(a) + var_b / len(b)
    se = math.sqrt(se2)

    numerator = se2 * se2
    denominator = (
        (var_a / len(a)) ** 2 / (len(a) - 1)
        + (var_b / len(b)) ** 2 / (len(b) - 1)
    )
    dfree = numerator / denominator

    lower_t = (estimate + margin) / se
    upper_t = (estimate - margin) / se
    p_lower = stats.t.sf(lower_t, dfree)
    p_upper = stats.t.cdf(upper_t, dfree)
    p_tost = max(p_lower, p_upper)

    critical_95 = stats.t.ppf(0.975, dfree)
    critical_90 = stats.t.ppf(0.95, dfree)

    ci95 = (
        estimate - critical_95 * se,
        estimate + critical_95 * se
    )
    ci90 = (
        estimate - critical_90 * se,
        estimate + critical_90 * se
    )

    return {
        "estimate": float(estimate),
        "ci_low": float(ci95[0]),
        "ci_high": float(ci95[1]),
        "ci90_low": float(ci90[0]),
        "ci90_high": float(ci90[1]),
        "statistic": np.nan,
        "df": float(dfree),
        "p_value": float(p_tost),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "n": f"{len(a)} + {len(b)}",
        "test": "TOST Welch equivalence test"
    }


def condition_summary(score_df):
    rows = []

    for mode, subset in score_df.groupby("mode", sort=False):
        if "participant" in score_df.columns:
            values = subset.groupby("participant")["score"].mean().values
            unit_label = "participants"
        else:
            values = subset["score"].values
            unit_label = "trials"

        mean, ci_low, ci_high = t_mean_ci(values)

        rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "n": len(values),
                "unit": unit_label,
                "mean_recalled": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high
            }
        )

    return pd.DataFrame(rows)


def position_summary(df, modes):
    rows = []

    for mode in modes:
        subset = df[df["mode"] == mode]

        if subset.empty:
            continue

        for position, group in subset.groupby("position"):
            successes = int(group["remembered"].sum())
            total = int(len(group))
            proportion = successes / total if total else np.nan
            low, high = wilson_ci(successes, total)

            rows.append(
                {
                    "mode": mode,
                    "position": int(position),
                    "recall_probability": proportion,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_items": total
                }
            )

    return pd.DataFrame(rows)



def _test_row(tests_df, analysis_name):
    match = tests_df[tests_df["analysis"] == analysis_name]
    if match.empty:
        return None
    return match.iloc[0]


def _condition_row(condition_df, mode):
    match = condition_df[condition_df["mode"] == mode]
    if match.empty:
        return None
    return match.iloc[0]


def _short_p(p):
    if pd.isna(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


def _significance_sentence(row):
    if row is None or pd.isna(row["p_value"]):
        return "Not enough data for a hypothesis test."

    if row["p_value"] < ALPHA:
        return "Statistically significant: reject H0."
    return "Not statistically significant: do not reject H0."


def _save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_free_recall_serial_position(position_df, output_path):
    subset = (
        position_df[position_df["mode"] == "free_recall_slow"]
        .sort_values("position")
        .copy()
    )

    if subset.empty:
        return

    x = subset["position"].to_numpy()
    y = subset["recall_probability"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(x, y, marker="o", linewidth=2.4, markersize=7)

    ax.set_title(
        "Free recall by serial position",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        "Primacy appears at the beginning and recency at the end. "
        "The separate summary plot tests these effects.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )

    ax.set_xlabel("Position in the 15-item list", fontsize=12)
    ax.set_ylabel("Items recalled (%)", fontsize=12)
    ax.set_xticks(range(1, 16))
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.22)

    ax.text(2, 96, "Beginning\npositions 1-3", ha="center", va="top", fontsize=10)
    ax.text(8, 96, "Middle\npositions 6-10", ha="center", va="top", fontsize=10)
    ax.text(14, 96, "End\npositions 13-15", ha="center", va="top", fontsize=10)

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.0f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    _save_figure(fig, output_path)



def plot_serial_recall_serial_position(position_df, output_path):
    subset = (
        position_df[position_df["mode"] == "serial_recall"]
        .sort_values("position")
        .copy()
    )

    if subset.empty:
        return

    x = subset["position"].to_numpy()
    y = subset["recall_probability"].to_numpy() * 100
    low = subset["ci95_low"].to_numpy() * 100
    high = subset["ci95_high"].to_numpy() * 100

    lower_error = np.maximum(y - low, 0)
    upper_error = np.maximum(high - y, 0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower_error, upper_error]),
        marker="o",
        linewidth=2.4,
        markersize=7,
        capsize=5
    )

    ax.set_title(
        "Serial recall rate by serial position",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        "Baseline serial recall. Error bars are 95% Wilson confidence intervals.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )

    ax.set_xlabel("Position in the sequence", fontsize=12)
    ax.set_ylabel("Correctly recalled (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.22)

    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.0f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    _save_figure(fig, output_path)


def serial_recall_rate_table(position_df):
    subset = (
        position_df[position_df["mode"] == "serial_recall"]
        .sort_values("position")
        .copy()
    )

    if subset.empty:
        return pd.DataFrame(
            columns=[
                "position",
                "recall_rate",
                "recall_percent",
                "ci95_low",
                "ci95_high",
                "n_items"
            ]
        )

    result = subset[
        ["position", "recall_probability", "ci95_low", "ci95_high", "n_items"]
    ].copy()
    result = result.rename(columns={"recall_probability": "recall_rate"})
    result["recall_percent"] = result["recall_rate"] * 100
    return result[
        [
            "position",
            "recall_rate",
            "recall_percent",
            "ci95_low",
            "ci95_high",
            "n_items"
        ]
    ]

def _region_values(df):
    baseline = df[df["mode"] == "free_recall_slow"].copy()
    if baseline.empty:
        return None

    group_cols = []
    if "participant" in baseline.columns:
        group_cols.append("participant")
    group_cols.append("_trial_id")

    rows = []
    for keys, group in baseline.groupby(group_cols, sort=False):
        row = {
            "Beginning (1-3)": group.loc[
                group["position"].between(1, 3), "remembered"
            ].mean(),
            "Middle (6-10)": group.loc[
                group["position"].between(6, 10), "remembered"
            ].mean(),
            "End (13-15)": group.loc[
                group["position"].between(13, 15), "remembered"
            ].mean()
        }

        if "participant" in baseline.columns:
            if isinstance(keys, tuple):
                row["participant"] = keys[0]
            else:
                row["participant"] = keys

        rows.append(row)

    result = pd.DataFrame(rows)

    if "participant" in baseline.columns:
        result = (
            result.groupby("participant", as_index=False)[
                ["Beginning (1-3)", "Middle (6-10)", "End (13-15)"]
            ]
            .mean()
        )

    return result


def plot_primacy_recency_summary(df, tests_df, output_path):
    region_df = _region_values(df)
    if region_df is None or region_df.empty:
        return

    labels = ["Beginning (1-3)", "Middle (6-10)", "End (13-15)"]
    means = []
    lows = []
    highs = []

    for label in labels:
        mean, low, high = t_mean_ci(region_df[label].to_numpy())
        means.append(mean * 100)
        lows.append(low * 100)
        highs.append(high * 100)

    lower = np.array(means) - np.array(lows)
    upper = np.array(highs) - np.array(means)

    primacy = _test_row(tests_df, "Free recall: primacy effect")
    recency = _test_row(tests_df, "Free recall: recency effect")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(3)

    bars = ax.bar(x, means, width=0.58)
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        capsize=7,
        linewidth=1.8
    )

    ax.set_title(
        "Primacy and recency in free recall",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        "Bars show mean recall in the beginning, middle, and end of the list. "
        "Error bars are 95% confidence intervals.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )

    ax.set_ylabel("Items recalled (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.22)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    annotation = (
        f"Primacy: beginning vs middle, {_short_p(primacy['p_value']) if primacy is not None else 'p = NA'}"
        f" | Recency: end vs middle, {_short_p(recency['p_value']) if recency is not None else 'p = NA'}"
    )
    ax.text(
        0.5,
        -0.15,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11
    )

    _save_figure(fig, output_path)


def plot_difference_from_baseline(
    condition_df,
    tests_df,
    baseline_mode,
    comparison_mode,
    analysis_name,
    title,
    comparison_label,
    output_path
):
    baseline = _condition_row(condition_df, baseline_mode)
    comparison = _condition_row(condition_df, comparison_mode)
    test = _test_row(tests_df, analysis_name)

    if baseline is None or comparison is None or test is None:
        return

    estimate = float(test["estimate"])
    low = float(test["ci95_low"])
    high = float(test["ci95_high"])

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    ax.axvline(0, linestyle="--", linewidth=1.8)
    ax.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="o",
        markersize=10,
        capsize=8,
        linewidth=2.2
    )

    span = max(abs(low), abs(high), 1.0)
    margin = max(0.7, span * 0.35)
    ax.set_xlim(-span - margin, span + margin)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])

    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.text(
        0.5,
        1.01,
        f"Baseline: {baseline['mean_recalled']:.2f}/15 | "
        f"{comparison_label}: {comparison['mean_recalled']:.2f}/15 | "
        f"Difference: {estimate:+.2f} | 95% CI [{low:.2f}, {high:.2f}] | "
        f"{_short_p(test['p_value'])}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5
    )

    ax.set_xlabel(
        f"Difference in items recalled ({comparison_label} - baseline)",
        fontsize=12
    )
    ax.grid(axis="x", alpha=0.22)

    direction = (
        f"{comparison_label} recalled {abs(estimate):.2f} fewer items"
        if estimate < 0
        else f"{comparison_label} recalled {abs(estimate):.2f} more items"
    )

    ci_statement = (
        "The 95% CI does not include 0, so the data support a difference."
        if low > 0 or high < 0
        else "The 95% CI includes 0, so the data do not clearly support a difference."
    )

    ax.text(
        0.5,
        -0.18,
        f"{direction}. {ci_statement} {_significance_sentence(test)}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5
    )

    ax.text(
        0,
        0.38,
        "No difference",
        ha="center",
        va="center",
        fontsize=9.5
    )

    _save_figure(fig, output_path)


def plot_serial_capacity(score_df, tests_df, output_path):
    subset = score_df[score_df["mode"] == "serial_recall"].copy()
    test = _test_row(tests_df, "Serial recall: limited working memory capacity")

    if subset.empty or test is None:
        return

    if "participant" in score_df.columns:
        values = subset.groupby("participant")["score"].mean().to_numpy()
        unit = "participants"
    else:
        values = subset["score"].to_numpy()
        unit = "trials"

    mean, low, high = t_mean_ci(values)
    list_length = int(round(subset["list_length"].median()))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.errorbar(
        0,
        mean,
        yerr=np.array([[mean - low], [high - mean]]),
        fmt="o",
        markersize=12,
        capsize=9,
        linewidth=2.3
    )
    ax.axhline(list_length, linestyle="--", linewidth=1.8)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, list_length + 1)
    ax.set_xticks([])
    ax.set_ylabel(f"Items recalled out of {list_length}", fontsize=12)
    ax.set_title(
        "Baseline serial recall capacity",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        f"Mean across {len(values)} {unit}. Error bar = 95% confidence interval.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )
    ax.grid(axis="y", alpha=0.22)

    ax.text(
        0.06,
        list_length - 0.15,
        f"Full list = {list_length}",
        va="top",
        fontsize=10
    )
    ax.text(
        0,
        mean - 1.4,
        f"Mean = {mean:.2f}\n95% CI [{low:.2f}, {high:.2f}]\n"
        f"{_short_p(test['p_value'])}",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.10)
    )

    _save_figure(fig, output_path)


def plot_error_types_clear(error_df, tests_df, output_path):
    if error_df.empty:
        return

    error_df = error_df.copy().sort_values("proportion", ascending=True)
    labels = error_df["error_type"].tolist()
    proportions = error_df["proportion"].to_numpy() * 100
    lows = error_df["ci_low"].to_numpy() * 100
    highs = error_df["ci_high"].to_numpy() * 100
    counts = error_df["count"].to_numpy()

    lower = proportions - lows
    upper = highs - proportions

    test = _test_row(tests_df, "Serial recall: error-type distribution")

    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(labels))

    bars = ax.barh(y, proportions, height=0.56)
    ax.errorbar(
        proportions,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        capsize=6,
        linewidth=1.6
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Share of all serial-recall errors (%)", fontsize=12)
    ax.set_title(
        "What kinds of errors occur in serial recall?",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        "Bars show the percentage of all errors. Error bars are 95% confidence intervals.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )

    ax.set_xlim(0, max(100, float(np.nanmax(highs)) + 8))
    ax.grid(axis="x", alpha=0.22)

    for bar, percentage, count in zip(bars, proportions, counts):
        ax.text(
            percentage + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%  (n={int(count)})",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

    if test is not None:
        ax.text(
            0.5,
            -0.13,
            f"Error types are tested against an equal-frequency null model: "
            f"{_short_p(test['p_value'])}. {_significance_sentence(test)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5
        )

    _save_figure(fig, output_path)


def plot_finger_tapping_equivalence(score_df, output_path):
    result = equivalence_tost(
        score_df,
        "serial_recall_finger_tapping",
        "serial_recall",
        FINGER_TAPPING_EQUIVALENCE_MARGIN
    )

    if result is None:
        return

    estimate = result["estimate"]
    low = result["ci90_low"]
    high = result["ci90_high"]
    margin = FINGER_TAPPING_EQUIVALENCE_MARGIN

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    ax.axvline(-margin, linestyle=":", linewidth=2)
    ax.axvline(0, linestyle="--", linewidth=1.7)
    ax.axvline(margin, linestyle=":", linewidth=2)

    ax.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="o",
        markersize=10,
        capsize=8,
        linewidth=2.2
    )

    span = max(abs(low), abs(high), margin) + 1
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.22)

    ax.set_title(
        "Finger tapping equivalence test",
        fontsize=18,
        fontweight="bold",
        pad=16
    )
    ax.text(
        0.5,
        1.01,
        f"Equivalence range: -{margin:.0f} to +{margin:.0f} item | "
        f"Estimate: {estimate:+.2f} | 90% CI [{low:.2f}, {high:.2f}] | "
        f"equivalence {_short_p(result['p_value'])}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5
    )
    ax.set_xlabel(
        "Difference in items recalled (finger tapping - baseline)",
        fontsize=12
    )

    ax.text(-margin, -0.47, f"-{margin:.0f}", ha="center", fontsize=10)
    ax.text(0, -0.47, "0", ha="center", fontsize=10)
    ax.text(margin, -0.47, f"+{margin:.0f}", ha="center", fontsize=10)

    conclusion = (
        "Equivalent within the chosen margin"
        if result["p_value"] < ALPHA
        else "Equivalence is not established"
    )

    inside_margin = low > -margin and high < margin
    margin_explanation = (
        "The full 90% CI is inside the equivalence range."
        if inside_margin
        else "The full 90% CI is not inside the equivalence range."
    )

    ax.text(
        0.5,
        -0.18,
        f"{margin_explanation} {conclusion}.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5
    )

    _save_figure(fig, output_path)


def add_result(results, name, h0, h1, result, notes=""):
    if result is None:
        results.append(
            {
                "analysis": name,
                "H0": h0,
                "H1": h1,
                "test": "Not enough data",
                "n": "",
                "estimate": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "decision": "Not enough data",
                "notes": notes
            }
        )
        return

    results.append(
        {
            "analysis": name,
            "H0": h0,
            "H1": h1,
            "test": result["test"],
            "n": result["n"],
            "estimate": result["estimate"],
            "ci95_low": result["ci_low"],
            "ci95_high": result["ci_high"],
            "statistic": result["statistic"],
            "df": result["df"],
            "p_value": result["p_value"],
            "decision": decision_from_p(result["p_value"]),
            "notes": notes
        }
    )


def run_all_tests(df, score_df):
    results = []

    primacy = region_effect_test(df, "primacy")
    add_result(
        results,
        "Free recall: primacy effect",
        "H0: There is no primacy effect. Recall for the first 3 items is not higher than recall for the middle items.",
        "H1: There is a primacy effect. Recall for the first 3 items is higher than recall for the middle items.",
        primacy,
        "Estimate is the difference in recall probability: first 3 minus middle."
    )

    recency = region_effect_test(df, "recency")
    add_result(
        results,
        "Free recall: recency effect",
        "H0: There is no recency effect. Recall for the last 3 items is not higher than recall for the middle items.",
        "H1: There is a recency effect. Recall for the last 3 items is higher than recall for the middle items.",
        recency,
        "Estimate is the difference in recall probability: last 3 minus middle."
    )

    fast = compare_modes(
        score_df,
        "free_recall_fast",
        "free_recall_slow"
    )
    add_result(
        results,
        "Free recall: faster presentation rate",
        "H0: Presentation speed has no effect on recall. Mean recall is the same for fast and slow presentation.",
        "H1: Presentation speed affects recall. Mean recall differs between fast and slow presentation.",
        fast,
        "Estimate is fast minus slow, in recalled items."
    )

    wm = compare_modes(
        score_df,
        "free_recall_working_memory",
        "free_recall_slow"
    )
    add_result(
        results,
        "Free recall: post-sequence working memory task",
        "H0: The working-memory task has no effect on recall. Mean recall is the same with and without the task.",
        "H1: The working-memory task affects recall. Mean recall differs when the task is added.",
        wm,
        "Estimate is working memory task minus slow baseline."
    )

    pause = compare_modes(
        score_df,
        "free_recall_pause",
        "free_recall_slow"
    )
    add_result(
        results,
        "Free recall: post-sequence pause",
        "H0: The pause has no effect on recall. Mean recall is the same with and without the pause.",
        "H1: The pause affects recall. Mean recall differs when a pause is inserted before recall.",
        pause,
        "Estimate is pause minus slow baseline."
    )

    capacity = serial_capacity_test(score_df)
    if capacity is not None:
        list_length = capacity["null_mean"]
        h0_capacity = f"H0: There is no evidence of limited serial-recall capacity. Mean recall equals the full list length of {list_length} items."
        h1_capacity = f"H1: Serial-recall capacity is limited. Mean recall is less than the full list length of {list_length} items."
    else:
        h0_capacity = "H0: There is no evidence of limited serial-recall capacity. Mean recall equals the full list length."
        h1_capacity = "H1: Serial-recall capacity is limited. Mean recall is below the full list length."

    add_result(
        results,
        "Serial recall: limited working memory capacity",
        h0_capacity,
        h1_capacity,
        capacity,
        "The confidence interval describes the mean baseline serial recall score."
    )

    error_df = classify_serial_errors(df)
    error_test = error_type_chi_square(error_df)
    add_result(
        results,
        "Serial recall: error-type distribution",
        "H0: The different serial-recall error types occur equally often. There is no evidence that one error type is more common than another.",
        "H1: The serial-recall error types do not occur equally often. At least one error type is more or less common than expected.",
        error_test,
        "See error_types.csv for counts, proportions, and 95% Wilson confidence intervals."
    )

    chunking = compare_modes(
        score_df,
        "serial_recall_chunking",
        "serial_recall"
    )
    add_result(
        results,
        "Serial recall: chunking",
        "H0: Chunking has no effect on serial recall. Mean recall is the same with and without chunking.",
        "H1: Chunking affects serial recall. Mean recall differs when chunking is used.",
        chunking,
        "Estimate is chunking minus baseline."
    )

    suppression_mode = None
    if (score_df["mode"] == "serial_recall_articulatory_suppresion").any():
        suppression_mode = "serial_recall_articulatory_suppresion"
    elif (score_df["mode"] == "serial_recall_articulatory_suppression").any():
        suppression_mode = "serial_recall_articulatory_suppression"

    suppression = None
    if suppression_mode:
        suppression = compare_modes(
            score_df,
            suppression_mode,
            "serial_recall"
        )

    add_result(
        results,
        "Serial recall: articulatory suppression",
        "H0: Articulatory suppression has no effect on serial recall. Mean recall is the same with and without suppression.",
        "H1: Articulatory suppression affects serial recall. Mean recall differs when suppression is used.",
        suppression,
        "Estimate is articulatory suppression minus baseline."
    )

    finger = compare_modes(
        score_df,
        "serial_recall_finger_tapping",
        "serial_recall"
    )
    add_result(
        results,
        "Serial recall: finger tapping difference test",
        "H0: Finger tapping has no effect on serial recall. Mean recall is the same with and without finger tapping.",
        "H1: Finger tapping affects serial recall. Mean recall differs when finger tapping is added.",
        finger,
        "Estimate is finger tapping minus baseline."
    )

    tost = equivalence_tost(
        score_df,
        "serial_recall_finger_tapping",
        "serial_recall",
        FINGER_TAPPING_EQUIVALENCE_MARGIN
    )

    if tost is None:
        results.append(
            {
                "analysis": "Serial recall: finger tapping equivalence test",
                "H0": "H0: Finger tapping changes recall by at least 1 item in either direction, so the effect is large enough to matter under the chosen equivalence margin.",
                "H1": "H1: Finger tapping changes recall by less than 1 item in either direction, so the two conditions are practically equivalent under the chosen margin.",
                "test": "Not enough data",
                "n": "",
                "estimate": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "decision": "Not enough data",
                "notes": ""
            }
        )
    else:
        equivalence_decision = (
            "Reject equivalence H0: the finger-tapping effect is statistically small enough to be considered equivalent within +/- 1 item"
            if tost["p_value"] < ALPHA
            else "Do not reject equivalence H0: the data do not establish that the finger-tapping effect is smaller than +/- 1 item"
        )

        results.append(
            {
                "analysis": "Serial recall: finger tapping equivalence test",
                "H0": "H0: Finger tapping changes recall by at least 1 item in either direction, so the effect is large enough to matter under the chosen equivalence margin.",
                "H1": "H1: Finger tapping changes recall by less than 1 item in either direction, so the two conditions are practically equivalent under the chosen margin.",
                "test": tost["test"],
                "n": tost["n"],
                "estimate": tost["estimate"],
                "ci95_low": tost["ci_low"],
                "ci95_high": tost["ci_high"],
                "statistic": np.nan,
                "df": tost["df"],
                "p_value": tost["p_value"],
                "decision": equivalence_decision,
                "notes": (
                    f"TOST margin = +/- {FINGER_TAPPING_EQUIVALENCE_MARGIN:.1f} item. "
                    f"90% CI = [{tost['ci90_low']:.3f}, {tost['ci90_high']:.3f}]. "
                    f"Lower one-sided p = {tost['p_lower']:.4f}; "
                    f"upper one-sided p = {tost['p_upper']:.4f}."
                )
            }
        )

    return pd.DataFrame(results), error_df


def write_report(
    df,
    score_df,
    test_results,
    condition_df,
    error_df,
    output_path
):
    has_participants = "participant" in df.columns

    lines = []
    lines.append("MEMORY EXPERIMENT ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    if has_participants:
        participant_count = df["participant"].nunique()
        lines.append(
            f"Participant IDs detected: {participant_count}. "
            "Condition tests use participant-level condition means and paired tests when possible."
        )
    else:
        lines.append(
            "No participant ID column was detected. Condition comparisons therefore treat "
            "individual trials as independent observations. If these trials came from repeated "
            "measurements of the same people, select the original participant CSV files together "
            "in the program, or add a participant column, for the statistically preferred analysis."
        )

    lines.append("")
    lines.append(f"Rows loaded: {len(df)}")
    lines.append(f"Trials detected: {len(score_df)}")
    lines.append(f"Alpha: {ALPHA}")
    lines.append(f"Confidence intervals: {int(CONFIDENCE * 100)}%")
    lines.append("")

    lines.append("CONDITION MEANS")
    lines.append("-" * 80)

    for _, row in condition_df.iterrows():
        lines.append(
            f"{row['label']}: mean = {row['mean_recalled']:.3f}, "
            f"95% CI [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}], "
            f"n = {row['n']} {row['unit']}"
        )

    lines.append("")
    lines.append("HYPOTHESIS TESTS")
    lines.append("-" * 80)

    for _, row in test_results.iterrows():
        lines.append("")
        lines.append(row["analysis"])
        lines.append(str(row["H0"]))
        lines.append(str(row["H1"]))
        lines.append(f"Test: {row['test']}")
        lines.append(f"n: {row['n']}")

        if pd.notna(row["estimate"]):
            lines.append(f"Estimate: {row['estimate']:.4f}")

        if pd.notna(row["ci95_low"]) and pd.notna(row["ci95_high"]):
            lines.append(
                f"95% CI: [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            )

        if pd.notna(row["statistic"]):
            lines.append(f"Test statistic: {row['statistic']:.4f}")

        if pd.notna(row["df"]):
            lines.append(f"df: {row['df']:.3f}")

        lines.append(f"p-value: {format_p(row['p_value'])}")
        lines.append(f"Decision: {row['decision']}")

        if str(row["notes"]).strip():
            lines.append(f"Notes: {row['notes']}")

    lines.append("")
    lines.append("SERIAL RECALL ERROR TYPES")
    lines.append("-" * 80)

    for _, row in error_df.iterrows():
        lines.append(
            f"{row['error_type']}: count = {int(row['count'])}, "
            f"proportion = {row['proportion']:.4f}, "
            f"95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_files(paths):
    df = load_csv_files(paths)
    df = add_trial_ids(df)

    score_df = trial_scores(df)
    tests_df, error_df = run_all_tests(df, score_df)
    condition_df = condition_summary(score_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(paths[0]).resolve().parent
    output_dir = base_dir / f"memory_analysis_output_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_df.to_csv(output_dir / "condition_summary.csv", index=False)
    tests_df.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    error_df.to_csv(output_dir / "error_types.csv", index=False)

    free_modes = [
        "free_recall_slow",
        "free_recall_fast",
        "free_recall_working_memory",
        "free_recall_pause"
    ]
    serial_modes = [
        "serial_recall",
        "serial_recall_chunking",
        "serial_recall_articulatory_suppresion",
        "serial_recall_articulatory_suppression",
        "serial_recall_finger_tapping"
    ]

    free_position = position_summary(df, free_modes)
    serial_position = position_summary(df, serial_modes)

    free_position.to_csv(
        output_dir / "free_recall_serial_position_summary.csv",
        index=False
    )
    serial_position.to_csv(
        output_dir / "serial_recall_position_summary.csv",
        index=False
    )

    serial_rate = serial_recall_rate_table(serial_position)
    serial_rate.to_csv(
        output_dir / "serial_recall_rate_by_position.csv",
        index=False
    )

    if not free_position.empty:
        plot_free_recall_serial_position(
            free_position,
            output_dir / "01_free_recall_serial_position.png"
        )

    plot_primacy_recency_summary(
        df,
        tests_df,
        output_dir / "02_free_recall_primacy_recency.png"
    )

    plot_difference_from_baseline(
        condition_df,
        tests_df,
        "free_recall_slow",
        "free_recall_fast",
        "Free recall: faster presentation rate",
        "Effect of faster presentation on free recall",
        "Fast presentation",
        output_dir / "03_free_recall_fast_vs_slow.png"
    )

    plot_difference_from_baseline(
        condition_df,
        tests_df,
        "free_recall_slow",
        "free_recall_working_memory",
        "Free recall: post-sequence working memory task",
        "Effect of a working-memory task on free recall",
        "Working-memory task",
        output_dir / "04_free_recall_working_memory.png"
    )

    plot_difference_from_baseline(
        condition_df,
        tests_df,
        "free_recall_slow",
        "free_recall_pause",
        "Free recall: post-sequence pause",
        "Effect of a pause before free recall",
        "Pause",
        output_dir / "05_free_recall_pause.png"
    )

    plot_serial_capacity(
        score_df,
        tests_df,
        output_dir / "06_serial_recall_capacity.png"
    )

    plot_error_types_clear(
        error_df,
        tests_df,
        output_dir / "07_serial_recall_error_types.png"
    )

    plot_difference_from_baseline(
        condition_df,
        tests_df,
        "serial_recall",
        "serial_recall_chunking",
        "Serial recall: chunking",
        "Effect of chunking on serial recall",
        "Chunking",
        output_dir / "08_serial_recall_chunking.png"
    )

    suppression_mode = None
    if (condition_df["mode"] == "serial_recall_articulatory_suppresion").any():
        suppression_mode = "serial_recall_articulatory_suppresion"
    elif (condition_df["mode"] == "serial_recall_articulatory_suppression").any():
        suppression_mode = "serial_recall_articulatory_suppression"

    if suppression_mode:
        plot_difference_from_baseline(
            condition_df,
            tests_df,
            "serial_recall",
            suppression_mode,
            "Serial recall: articulatory suppression",
            "Effect of articulatory suppression on serial recall",
            "Articulatory suppression",
            output_dir / "09_serial_recall_articulatory_suppression.png"
        )

    plot_difference_from_baseline(
        condition_df,
        tests_df,
        "serial_recall",
        "serial_recall_finger_tapping",
        "Serial recall: finger tapping difference test",
        "Effect of finger tapping on serial recall",
        "Finger tapping",
        output_dir / "10_serial_recall_finger_tapping.png"
    )

    plot_finger_tapping_equivalence(
        score_df,
        output_dir / "11_finger_tapping_equivalence.png"
    )

    if not serial_rate.empty:
        plot_serial_recall_serial_position(
            serial_position,
            output_dir / "12_serial_recall_rate_by_position.png"
        )

    write_report(
        df,
        score_df,
        tests_df,
        condition_df,
        error_df,
        output_dir / "analysis_report.txt"
    )

    return output_dir, tests_df


def _clean_input_path(value):
    return value.strip().strip('"').strip("'")


def _collect_csv_paths_from_prompt():
    print("Memory Experiment Analysis")
    print("Enter the path to your experiment CSV file(s), one per line.")
    print("Do NOT enter the path to this .py program.")
    print("Press Enter on an empty line when finished.\n")

    paths = []
    while True:
        value = input(f"Experiment CSV {len(paths) + 1}: ").strip()
        if not value:
            break

        path_text = _clean_input_path(value)
        path = Path(path_text)

        if path.suffix.lower() != ".csv":
            print("  Not a CSV file. Please enter a .csv experiment-data file, not the .py program.\n")
            continue

        if not path.is_file():
            print(f"  File not found: {path_text}\n")
            continue

        paths.append(path_text)

    return paths


def _print_test_summary(tests_df):
    print("\nHypothesis-test summary")
    print("=" * 80)

    for _, row in tests_df.iterrows():
        print(row["analysis"])
        print(f"  p = {format_p(row['p_value'])}")
        print(f"  {row['decision']}")

        if pd.notna(row["estimate"]):
            print(f"  Estimate = {row['estimate']:.3f}")

        if pd.notna(row["ci95_low"]) and pd.notna(row["ci95_high"]):
            print(f"  95% CI = [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]")

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Analyze one merged memory-experiment CSV or multiple participant CSV files. "
            "All results and plots are saved to a new output folder."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="CSV file(s) to analyze. If omitted, the program asks for paths interactively."
    )
    args = parser.parse_args()

    paths = [_clean_input_path(p) for p in args.csv_files]
    if not paths:
        paths = _collect_csv_paths_from_prompt()

    if not paths:
        print("No CSV files supplied.")
        return 1

    wrong_type = [p for p in paths if Path(p).suffix.lower() != ".csv"]
    if wrong_type:
        print("\nThese are not CSV data files:")
        for path in wrong_type:
            print(f"  {path}")
        print("\nUse the path to your experiment data, for example memory_results_merged.csv, not this .py program.")
        return 1

    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        print("\nThe following file(s) were not found:")
        for path in missing:
            print(f"  {path}")
        return 1

    try:
        output_dir, tests_df = analyze_files(paths)
    except Exception as exc:
        print(f"\nAnalysis failed: {exc}")
        return 1

    _print_test_summary(tests_df)
    print(f"Output folder: {output_dir}")
    print("Created CSV summaries, serial-recall rates by position, hypothesis tests, confidence intervals, plots, and analysis_report.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
