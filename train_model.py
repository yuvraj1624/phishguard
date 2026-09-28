"""
train_model.py
==============
This script does the whole machine learning job, in the correct order.

THE MACHINE LEARNING WORKFLOW IN SIMPLE WORDS
---------------------------------------------
Step 1  LOAD      read data/dataset.csv (url + label)
Step 2  CLEAN     remove duplicates and broken rows
Step 3  FEATURES  turn every URL into 18 numbers
Step 4  LABELS    make PHISHING the number 1 (the "positive" class)
Step 5  SPLIT     80% to train, 20% to test  (done BEFORE anything is fitted)
Step 6  TRAIN     fit Logistic Regression, Decision Tree, Random Forest
Step 7  MEASURE   accuracy, precision, recall, F1, confusion matrix
Step 8  CHOOSE    keep the model with the best F1 on the phishing class
Step 9  SAVE      write the winner to models/phishguard_model.joblib
Step 10  GRAPHS    draw the charts into results/

WHY IS STEP 5 SO IMPORTANT?
---------------------------
The test set is data the model has NEVER seen. If we measured on data the
model was trained on, every model would look almost perfect and the numbers
would be useless. Splitting at step 5 and never training on the test part is
what makes the results honest.

HOW TO RUN:
    python train_model.py
    python train_model.py --sample 20000     (faster, for testing only)
"""

import argparse
import datetime
import os
import time
import warnings

# A harmless version-mismatch warning between scikit-learn and scipy.
# It does not affect the results, so we hide it to keep the output clean.
warnings.filterwarnings("ignore", message="Unknown solver options")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Matplotlib needs this line so it draws into a file instead of trying to
# open a window. It must come BEFORE we import pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import (FEATURE_DESCRIPTIONS, FEATURE_NAMES,
                                features_to_dataframe)

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "dataset.csv")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")
MODEL_FILE = os.path.join(MODEL_FOLDER, "phishguard_model.joblib")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

# The one setting that makes the whole run repeatable.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Names we use on screen, so nobody has to remember which number means what.
CLASS_NAMES = ["Legitimate", "Phishing"]

# The four numbers we actually score. Anything else in a result dictionary
# (the confusion matrix, the raw predictions) is kept for the graphs only.
METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]


# ===========================================================================
# STEP 1 + 2 : LOAD AND CLEAN
# ===========================================================================
def load_dataset():
    """Read data/dataset.csv and remove any remaining duplicates."""
    if not os.path.exists(DATA_FILE):
        print()
        print("ERROR: data/dataset.csv was not found.")
        print("Run this first:   python download_dataset.py")
        raise SystemExit(1)

    print("=" * 70)
    print("STEP 1-2 : LOADING AND CLEANING THE DATASET")
    print("=" * 70)

    dataset = pd.read_csv(DATA_FILE)

    # Safety net: the download script already removed these, but we check
    # again so the training script can be run on its own.
    rows_at_start = len(dataset)
    dataset = dataset.dropna(subset=["url", "label"])
    dataset["url"] = dataset["url"].astype(str).str.strip()
    dataset = dataset[dataset["url"].str.len() > 0]
    dataset = dataset[dataset["label"].isin([0, 1])]
    dataset = dataset.drop_duplicates(subset=["url"], keep="first")
    dataset = dataset.reset_index(drop=True)

    print(f"    Rows in data/dataset.csv   : {rows_at_start:,}")
    print(f"    Rows after cleaning        : {len(dataset):,}")
    print(f"    Rows removed while cleaning: "
          f"{rows_at_start - len(dataset):,}")

    # The original dataset uses  label 1 = legitimate, 0 = phishing.
    # For the whole project we flip it so that PHISHING is class 1.
    # Why? Because in every formula we use, class 1 is the one we care
    # about catching. This way "recall" means
    # "out of all the real phishing sites, how many did we catch?".
    dataset["is_phishing"] = (dataset["label"] == 0).astype(int)

    print()
    print("    Class balance (real counts read from the file):")
    for value, name in enumerate(CLASS_NAMES):
        count = int((dataset["is_phishing"] == value).sum())
        share = count / len(dataset) * 100
        print(f"        {name:<12} {count:>8,}   ({share:5.2f}%)")

    return dataset


# ===========================================================================
# STEP 3 : FEATURE EXTRACTION
# ===========================================================================
def build_feature_table(dataset):
    """Turn the url column into a table of 18 numbers per URL."""
    print()
    print("=" * 70)
    print("STEP 3 : EXTRACTING FEATURES")
    print("=" * 70)
    print(f"    Turning {len(dataset):,} URLs into "
          f"{len(FEATURE_NAMES)} numbers each...")

    start = time.time()
    feature_table = features_to_dataframe(
        dataset["url"].tolist(), show_progress=True
    )
    elapsed = time.time() - start

    print(f"    Feature table shape        : {feature_table.shape}")
    print(f"    Time taken                 : {elapsed:.1f} seconds")

    return feature_table


# ===========================================================================
# STEP 5 : SPLIT
# ===========================================================================
def split_data(feature_table, dataset):
    """
    Split into training and test parts.

    random_state=42 makes this split identical every single time you run it.
    stratify keeps the same class balance in both halves.
    """
    print()
    print("=" * 70)
    print("STEP 5 : SPLITTING INTO TRAINING AND TEST SETS")
    print("=" * 70)

    features = feature_table.to_numpy()
    labels = dataset["is_phishing"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"    Training set : {len(x_train):,} rows "
          f"({len(x_train) / len(features) * 100:.0f}%)")
    print(f"    Test set     : {len(x_test):,} rows "
          f"({len(x_test) / len(features) * 100:.0f}%)")
    print(f"    random_state : {RANDOM_STATE}  "
          f"(so this split is always the same)")
    print()
    print("    NO DATA LEAKAGE: the scaler and the models only ever see the")
    print("    training rows. The test rows are kept untouched until step 7.")

    return x_train, x_test, y_train, y_test


# ===========================================================================
# STEP 6 : THE THREE MODELS
# ===========================================================================
def build_models():
    """
    Return the three models we compare, in a dictionary.

    Why these three? (good viva answer)
    ------------------------------------
    * Logistic Regression - the simplest possible classifier. It draws one
      straight line between the classes. Easy to explain, very fast, and it
      gives a real probability. Good baseline.
    * Decision Tree - asks yes/no questions one after another ("is the URL
      longer than 40? then is the TLD risky?..."). Very easy to visualise.
    * Random Forest - 100 Decision Trees that each see a random part of the
      data, and they vote together. Fixes the main weakness of a single tree
      (it overfits) and is normally the strongest of the three.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000,
                                         random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ===========================================================================
# STEP 7 : MEASURE EVERY MODEL
# ===========================================================================
def measure(model, x_test, y_test):
    """
    Score one model on the test set.

    Remember: class 1 = Phishing. So every number below is about PHISHING.
    """
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "predictions": predictions,
    }


# ===========================================================================
# STEP 9 : SAVE THE WINNER
# ===========================================================================
def get_importances(model):
    """
    Try to work out how much the model relied on each feature.
    Returns a list of 18 numbers that add up to 1, or None if the model
    cannot tell us (in which case the app simply skips the "why" panel).
    """
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.array(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        # Logistic Regression has weights instead of importances.
        # We take the absolute size, because a big negative weight is
        # just as important as a big positive one.
        values = np.abs(np.array(estimator.coef_, dtype=float)).ravel()

    if values is None or values.sum() == 0:
        return None

    return (values / values.sum()).tolist()


def save_model(bundle):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)

    size_mb = os.path.getsize(MODEL_FILE) / 1048576
    print(f"    Saved to : {MODEL_FILE}")
    print(f"    Size     : {size_mb:.1f} MB")


# ===========================================================================
# STEP 10 : THE GRAPHS
# ===========================================================================
def style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)


def plot_class_distribution(dataset):
    """Bar chart: how many legitimate vs phishing URLs we have."""
    counts = [int((dataset["is_phishing"] == 0).sum()),
              int((dataset["is_phishing"] == 1).sum())]

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(CLASS_NAMES, counts,
                    color=["#2e7d32", "#c62828"], width=0.55)
    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, count,
                  f"{count:,}\n({count / sum(counts) * 100:.1f}%)",
                  ha="center", va="bottom", fontsize=11, fontweight="bold")

    axis.set_title("Class Distribution of the Dataset", fontsize=14,
                   fontweight="bold")
    axis.set_ylabel("Number of URLs")
    axis.set_ylim(0, max(counts) * 1.22)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_FOLDER, "class_distribution.png"),
                   dpi=150)
    plt.close(figure)


def plot_single_metric(all_metrics, metric, filename, title):
    """One bar chart comparing the same metric across all three models."""
    names = list(all_metrics.keys())
    values = [all_metrics[name][metric] for name in names]

    figure, axis = plt.subplots(figsize=(7.5, 5))
    bars = axis.bar(names, values, color="#1565c0", width=0.55)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                  f"{value:.4f}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold")

    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_ylim(0, 1.12)
    axis.tick_params(axis="x", rotation=12)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_FOLDER, filename), dpi=150)
    plt.close(figure)


def plot_all_metrics(all_metrics):
    """Grouped bar chart with all four metrics side by side."""
    names = list(all_metrics.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    colours = ["#1565c0", "#ef6c00", "#2e7d32", "#6a1b9a"]

    positions = np.arange(len(names))
    width = 0.2

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, (metric, label, colour) in enumerate(
            zip(metrics, labels, colours)):
        values = [all_metrics[name][metric] for name in names]
        offset = (index - 1.5) * width
        axis.bar(positions + offset, values, width,
                 label=label, color=colour, edgecolor="white")

    axis.set_xticks(positions)
    axis.set_xticklabels(names)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score (PHISHING is the positive class)")
    axis.set_title("Model Comparison - All Metrics", fontsize=14,
                   fontweight="bold")
    axis.legend(loc="lower right", frameon=False)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_FOLDER,
                                "all_metrics_comparison.png"), dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, title, filename):
    """Draw one confusion matrix as a heatmap with the real numbers."""
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        ax=axis,
    )
    axis.set_xlabel("Model predicted")
    axis.set_ylabel("Actual reality")
    axis.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Spell out the meaning of each cell so the chart is useful in a report.
    true_negative, false_positive = matrix[0][0], matrix[0][1]
    false_negative, true_positive = matrix[1][0], matrix[1][1]
    caption = (
        f"TP={true_positive:,}  (phishing caught)   "
        f"FN={false_negative:,}  (phishing MISSED)\n"
        f"TN={true_negative:,}  (legit allowed)   "
        f"FP={false_positive:,}  (false alarm)"
    )
    figure.text(0.5, 0.02, caption, ha="center", fontsize=9,
                bbox={"facecolor": "#f2f2f2", "edgecolor": "none",
                      "boxstyle": "round,pad=0.5"})

    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(os.path.join(RESULTS_FOLDER, filename), dpi=150)
    plt.close(figure)


def plot_predicted_vs_actual(best_name, best_result, y_test):
    """How many of each class were really there, vs how many we predicted."""
    actual_phishing = int((y_test == 1).sum())
    actual_legitimate = int((y_test == 0).sum())
    predicted_phishing = int((best_result["predictions"] == 1).sum())
    predicted_legitimate = int((best_result["predictions"] == 0).sum())

    labels = ["Legitimate", "Phishing"]
    actual = [actual_legitimate, actual_phishing]
    predicted = [predicted_legitimate, predicted_phishing]

    positions = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - width / 2, actual, width, label="Actual (test set)",
             color="#37474f")
    axis.bar(positions + width / 2, predicted, width,
             label=f"Predicted by {best_name}", color="#ef6c00")

    for offset, values in ((-width / 2, actual), (width / 2, predicted)):
        for position, value in zip(positions, values):
            axis.text(position + offset, value, f"{value:,}",
                      ha="center", va="bottom", fontsize=10)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Number of URLs in the test set")
    axis.set_title("Actual vs Predicted Class Counts", fontsize=14,
                   fontweight="bold")
    axis.legend(frameon=False)
    axis.set_ylim(0, max(actual + predicted) * 1.18)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_FOLDER, "predicted_vs_actual.png"),
                   dpi=150)
    plt.close(figure)


def plot_feature_importance(importances, model_name):
    """Which features did the model actually rely on the most?"""
    if importances is None:
        return

    pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1])
    names = [name for name, _ in pairs]
    values = [value for _, value in pairs]

    figure, axis = plt.subplots(figsize=(9, 6.5))
    colours = ["#c62828" if value > 0 else "#90a4ae" for value in values]
    axis.barh(names, values, color=colours)
    axis.set_xlabel("Feature importance (higher = the model used it more)")
    axis.set_title(f"Feature Importance - {model_name}", fontsize=14,
                   fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_FOLDER, "feature_importance.png"),
                   dpi=150)
    plt.close(figure)


def make_all_graphs(dataset, all_metrics, best_name, best_result, y_test,
                    importances):
    """Draw every chart and list what was created."""
    print()
    print("=" * 70)
    print("STEP 10 : DRAWING THE GRAPHS")
    print("=" * 70)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plot_class_distribution(dataset)
    plot_single_metric(all_metrics, "accuracy", "accuracy_comparison.png",
                       "Model Accuracy Comparison")
    plot_single_metric(all_metrics, "precision", "precision_comparison.png",
                       "Model Precision Comparison (PHISHING class)")
    plot_single_metric(all_metrics, "recall", "recall_comparison.png",
                       "Model Recall Comparison (PHISHING class)")
    plot_single_metric(all_metrics, "f1", "f1_comparison.png",
                       "Model F1-score Comparison (PHISHING class)")
    plot_all_metrics(all_metrics)

    for name, result in all_metrics.items():
        safe = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            result["confusion_matrix"],
            f"Confusion Matrix - {name}",
            f"confusion_matrix_{safe}.png",
        )

    plot_confusion_matrix(
        best_result["confusion_matrix"],
        f"Confusion Matrix - SELECTED MODEL ({best_name})",
        "confusion_matrix_best_model.png",
    )
    plot_predicted_vs_actual(best_name, best_result, y_test)
    plot_feature_importance(importances, best_name)

    created = sorted(f for f in os.listdir(RESULTS_FOLDER)
                     if f.endswith(".png"))
    for index, name in enumerate(created, start=1):
        print(f"    {index:>2}. results/{name}")
    print(f"    Total: {len(created)} graphs saved in results/")


# ===========================================================================
# THE REPORT FILES (real numbers only)
# ===========================================================================
def write_reports(dataset, all_metrics, best_name, timings, total_rows,
                  y_test, importances):
    """Save the dataset statistics, comparison table and full report."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # ---- dataset_statistics.csv -----------------------------------------
    phishing = int((dataset["is_phishing"] == 1).sum())
    legitimate = int((dataset["is_phishing"] == 0).sum())
    statistics = {
        "Total rows after cleaning": total_rows,
        "Legitimate URLs": legitimate,
        "Phishing URLs": phishing,
        "Legitimate share (%)": round(legitimate / total_rows * 100, 2),
        "Phishing share (%)": round(phishing / total_rows * 100, 2),
        "Number of features": len(FEATURE_NAMES),
        "Training rows": int(len(y_test) * 0 + (total_rows - len(y_test))),
        "Test rows": len(y_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    pd.DataFrame([statistics]).T.rename(columns={0: "value"}).to_csv(
        os.path.join(RESULTS_FOLDER, "dataset_statistics.csv")
    )

    # ---- model_comparison.csv -------------------------------------------
    rows = []
    for name, result in all_metrics.items():
        matrix = result["confusion_matrix"]
        rows.append({
            "Model": name,
            "Accuracy": round(result["accuracy"], 4),
            "Precision": round(result["precision"], 4),
            "Recall": round(result["recall"], 4),
            "F1-score": round(result["f1"], 4),
            "True Positives (phishing caught)": int(matrix[1][1]),
            "False Positives (false alarm)": int(matrix[0][1]),
            "False Negatives (phishing missed)": int(matrix[1][0]),
            "True Negatives (legit allowed)": int(matrix[0][0]),
            "Training time (seconds)": round(timings[name], 2),
            "Selected": "YES" if name == best_name else "no",
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        os.path.join(RESULTS_FOLDER, "model_comparison.csv"), index=False
    )

    # ---- best_model.txt --------------------------------------------------
    with open(os.path.join(RESULTS_FOLDER, "best_model.txt"), "w",
              encoding="utf-8") as file:
        file.write(f"SELECTED MODEL : {best_name}\n")
        file.write(f"Selected using  : F1-score of the PHISHING class\n")
        result = all_metrics[best_name]
        file.write(f"Accuracy        : {result['accuracy']:.4f}\n")
        file.write(f"Precision       : {result['precision']:.4f}\n")
        file.write(f"Recall          : {result['recall']:.4f}\n")
        file.write(f"F1-score        : {result['f1']:.4f}\n")
        file.write(f"Model file      : {MODEL_FILE}\n")

    # ---- metrics_report.txt ---------------------------------------------
    with open(os.path.join(RESULTS_FOLDER, "metrics_report.txt"), "w",
              encoding="utf-8") as file:
        file.write("=" * 72 + "\n")
        file.write("PhishGuard - Full Evaluation Report\n")
        file.write("All numbers below were calculated from the real dataset.\n")
        file.write("Positive class = PHISHING (1)\n")
        file.write("=" * 72 + "\n\n")

        file.write("DATASET\n")
        file.write("-" * 72 + "\n")
        for key, value in statistics.items():
            file.write(f"  {key:<28} : {value}\n")
        file.write(f"  {'random_state':<28} : {RANDOM_STATE}\n\n")

        file.write("FEATURES USED (" + str(len(FEATURE_NAMES)) + ")\n")
        file.write("-" * 72 + "\n")
        for index, name in enumerate(FEATURE_NAMES, start=1):
            file.write(f"  {index:>2}. {name:<22} "
                      f"{FEATURE_DESCRIPTIONS[name]}\n")
        file.write("\n")

        file.write("MODEL COMPARISON\n")
        file.write("-" * 72 + "\n")
        file.write(f"  {'Model':<22}{'Accuracy':>10}{'Precision':>11}"
                   f"{'Recall':>9}{'F1':>9}\n")
        for name, result in all_metrics.items():
            marker = " <== SELECTED" if name == best_name else ""
            file.write(
                f"  {name:<22}{result['accuracy']:>10.4f}"
                f"{result['precision']:>11.4f}"
                f"{result['recall']:>9.4f}{result['f1']:>9.4f}{marker}\n"
            )
        file.write("\n")

        file.write("CONFUSION MATRICES (rows = actual, columns = predicted)\n")
        file.write("-" * 72 + "\n")
        for name, result in all_metrics.items():
            matrix = result["confusion_matrix"]
            file.write(f"\n  {name}\n")
            file.write(f"    {'':<14}{'Pred Legitimate':>18}"
                       f"{'Pred Phishing':>16}\n")
            file.write(f"    {'Actual Legitimate':<14}{matrix[0][0]:>18,}"
                       f"{matrix[0][1]:>16,}\n")
            file.write(f"    {'Actual Phishing':<14}{matrix[1][0]:>18,}"
                       f"{matrix[1][1]:>16,}\n")
            file.write(f"    True Negatives  (legit allowed)  : {matrix[0][0]:,}\n")
            file.write(f"    False Positives (false alarm)    : {matrix[0][1]:,}\n")
            file.write(f"    False Negatives (phishing missed): {matrix[1][0]:,}\n")
            file.write(f"    True Positives  (phishing caught): {matrix[1][1]:,}\n")

        if importances is not None:
            file.write("\n\nFEATURE IMPORTANCE (selected model)\n")
            file.write("-" * 72 + "\n")
            ordered = sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: x[1], reverse=True)
            for name, value in ordered:
                file.write(f"  {name:<24}{value:>8.4f}\n")

    print()
    print("=" * 70)
    print("REPORT FILES WRITTEN TO results/")
    print("=" * 70)
    for name in ["dataset_statistics.csv", "model_comparison.csv",
                 "metrics_report.txt", "best_model.txt"]:
        path = os.path.join(RESULTS_FOLDER, name)
        if os.path.exists(path):
            print(f"    {name:<28} {os.path.getsize(path):>8,} bytes")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print()
    print("#" * 70)
    print("#  PhishGuard - Training the phishing URL detection models")
    print("#" * 70)

    parser = argparse.ArgumentParser(
        description="Train and compare 3 machine learning models."
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Use only this many rows. Faster for testing. 0 = use all.",
    )
    arguments = parser.parse_args()

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # ---- steps 1 and 2 ----------------------------------------------------
    dataset = load_dataset()
    total_rows = len(dataset)

    if arguments.sample and arguments.sample < total_rows:
        dataset = dataset.sample(n=arguments.sample, random_state=RANDOM_STATE)
        dataset = dataset.reset_index(drop=True)
        total_rows = len(dataset)
        print()
        print(f"    NOTE: --sample was used, so only {total_rows:,} rows "
              f"are being used.")

    # ---- step 3 -----------------------------------------------------------
    feature_table = build_feature_table(dataset)
    dataset = pd.concat([dataset.reset_index(drop=True),
                         feature_table.reset_index(drop=True)], axis=1)

    # ---- step 5 -----------------------------------------------------------
    x_train, x_test, y_train, y_test = split_data(feature_table, dataset)

    # ---- step 6 and 7 -----------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 6-7 : TRAINING AND MEASURING THE THREE MODELS")
    print("=" * 70)

    all_metrics = {}
    timings = {}

    for name, model in build_models().items():
        print(f"\n    Training {name} ...")
        start = time.time()
        model.fit(x_train, y_train)
        timings[name] = time.time() - start

        result = measure(model, x_test, y_test)
        all_metrics[name] = result

        matrix = result["confusion_matrix"]
        print(f"        done in {timings[name]:.1f} seconds")
        print(f"        Accuracy  : {result['accuracy']:.4f}")
        print(f"        Precision : {result['precision']:.4f}")
        print(f"        Recall    : {result['recall']:.4f}")
        print(f"        F1-score  : {result['f1']:.4f}")
        print(f"        Confusion matrix:")
        print(f"            actual legit  -> predicted legit "
              f"{matrix[0][0]:>7,}   predicted phishing {matrix[0][1]:>7,}")
        print(f"            actual phish  -> predicted legit "
              f"{matrix[1][0]:>7,}   predicted phishing {matrix[1][1]:>7,}")

    # ---- step 8 -----------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 8 : CHOOSING THE BEST MODEL")
    print("=" * 70)
    print("    Selection rule: the highest F1-score on the PHISHING class.")
    print("    Reason: F1 balances precision and recall. A phishing filter")
    print("    that is too careful misses real attacks, and one that is too")
    print("    careless blocks honest users. F1 punishes both mistakes.")
    print()
    print(f"    {'Model':<24}{'F1 (phishing)':>16}")
    for name, result in sorted(all_metrics.items(),
                               key=lambda x: x[1]["f1"], reverse=True):
        print(f"    {name:<24}{result['f1']:>16.4f}")

    best_name = max(all_metrics, key=lambda name: all_metrics[name]["f1"])
    best_result = all_metrics[best_name]

    print()
    print(f"    ==> SELECTED : {best_name}  "
          f"(F1 = {best_result['f1']:.4f})")

    # ---- step 9 -----------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 9 : SAVING THE MODEL")
    print("=" * 70)

    fitted_models = build_models()
    # Refit the winner on the training data only, then save it.
    final_model = fitted_models[best_name]
    final_model.fit(x_train, y_train)

    importances = get_importances(final_model)
    if importances is None:
        print("    (this model cannot report feature importances; the app "
              "will skip the 'why' panel)")

    bundle = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "importances": importances,
        "best_model_name": best_name,
        "metrics": {key: float(best_result[key]) for key in METRIC_KEYS},
        "all_metrics": {
            name: {key: float(result[key]) for key in METRIC_KEYS}
            for name, result in all_metrics.items()
        },
        "class_names": CLASS_NAMES,
        "phishing_is_class_1": True,
        "selection_metric": "F1-score of the PHISHING class",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_name": "PhiUSIIL Phishing URL (Website), UCI ML Repository",
        "training_rows": int(len(x_train)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_model(bundle)

    # ---- step 10 ----------------------------------------------------------
    make_all_graphs(dataset, all_metrics, best_name, best_result, y_test,
                    importances)
    write_reports(dataset, all_metrics, best_name, timings, total_rows,
                  y_test, importances)

    print()
    print("#" * 70)
    print("#  TRAINING COMPLETE")
    print("#" * 70)
    print(f"  Best model      : {best_name}")
    print(f"  F1 (phishing)   : {best_result['f1']:.4f}")
    print(f"  Model file      : {MODEL_FILE}")
    print(f"  Graphs          : {RESULTS_FOLDER}")
    print()
    print("  Next:  python evaluate_model.py")
    print("  Then:  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
