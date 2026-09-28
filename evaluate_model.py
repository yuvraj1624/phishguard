"""
evaluate_model.py
=================
What this script is FOR
-----------------------
train_model.py CREATES the model. This script CHECKS it.

It does three things:

  1. Loads the saved model from models/phishguard_model.joblib
     (this proves the file really works and is not corrupt)

  2. Re-runs the exact same test that training used, and checks that the
     numbers come out identical. If they do, the results are reproducible.
     This is the "did we really measure what we said we measured?" check.

  3. Runs a small set of example URLs through the model and prints the
     result for each one.

A VERY IMPORTANT WORDING POINT
-----------------------------
This script prints two separate lines for every example URL:

    Model prediction       : ...
    Confirmed website safety: NOT ESTABLISHED

That wording is deliberate. The model has only read the TEXT of a link.
It has never visited the website. So it can never confirm that a site is
genuinely safe. It only gives an opinion based on patterns.

HOW TO RUN:
    python evaluate_model.py
    python evaluate_model.py --url "https://example.com"
    python evaluate_model.py --skip-verify
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from feature_extraction import (FEATURE_DESCRIPTIONS, explain_features,
                                extract_features, features_to_vector)

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(PROJECT_FOLDER, "models", "phishguard_model.joblib")
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "dataset.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_NAMES = ["Legitimate", "Phishing"]

# Example URLs for the testing section. These are demonstration strings only.
# We have NOT visited any of them, and this project makes no claim about
# whether any real website is actually safe or dangerous.
#
# The list is deliberately honest. It contains URLs the model gets RIGHT,
# and also two that it gets WRONG, so the limitation is visible rather
# than hidden. See the note printed at the end.
EXAMPLE_URLS = [
    # --- ordinary websites, written the usual way ---
    "https://www.wikipedia.org/",
    "https://www.github.com/",
    "https://www.amazon.com/",
    "https://www.microsoft.com/",
    # --- obvious phishing links ---
    "http://192.168.0.1/login",
    "http://secure-login.paypal-verify-account.xyz/login?session=9981",
    "http://xn--pypal-4ve.com/signin",
    "http://my-bank.account-verify.tk/update.php?user=1024",
    # --- two the model gets WRONG, kept in on purpose ---
    "https://stackoverflow.com/",
    "https://en.wikipedia.org/wiki/Phishing",
]

HONESTY_NOTE = """
    A NOTE ON THE LAST TWO URLS
    ---------------------------
    'https://stackoverflow.com/' and 'https://en.wikipedia.org/wiki/Phishing'
    are ordinary, legitimate websites, and this model still calls them
    phishing. That is not a coding mistake and not a fluke. It is a real
    limitation of the DATASET, and we have chosen to show it rather than
    hide it.

    All 134,850 legitimate URLs in the PhiUSIIL dataset are bare
    'https://www.something.tld' homepages. Not one lacks a 'www.', and not
    one has a page after the domain name. A model trained only on that
    learns two shortcuts:

        "no www.  ->  phishing"        and        "any path  ->  phishing"

    Those rules fit the dataset perfectly, which is exactly why the score
    looks high. They simply do not describe phishing in the real world.

    Run   python robustness_check.py   to reproduce this finding.
"""


def load_bundle():
    """Load the saved model bundle. Fails loudly if the file is missing."""
    print()
    print("=" * 70)
    print("STEP 1 : LOADING THE SAVED MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE):
        print()
        print("ERROR: no model file found at:")
        print("      ", MODEL_FILE)
        print()
        print("Train it first:   python train_model.py")
        raise SystemExit(1)

    bundle = joblib.load(MODEL_FILE)

    print(f"    File              : {MODEL_FILE}")
    print(f"    Size              : "
          f"{os.path.getsize(MODEL_FILE) / 1048576:.1f} MB")
    print(f"    Selected model    : {bundle['best_model_name']}")
    print(f"    Trained on        : {bundle['trained_at']}")
    print(f"    scikit-learn      : {bundle['sklearn_version']}")
    print(f"    Training rows     : {bundle['training_rows']:,}")
    print(f"    Dataset           : {bundle['dataset_name']}")
    print(f"    Selection metric  : {bundle['selection_metric']}")
    print()
    print("    Metrics recorded when the model was trained:")
    for key, value in bundle["metrics"].items():
        print(f"        {key:<12} : {value:.4f}")

    return bundle


def verify_metrics(bundle):
    """
    Rebuild the exact same train/test split and re-measure the selected
    model. If our numbers match the stored ones, everything is consistent.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split

    from feature_extraction import features_to_dataframe

    print()
    print("=" * 70)
    print("STEP 2 : VERIFYING THE RESULTS ARE REPRODUCIBLE")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print("    data/dataset.csv not found - skipping this check.")
        return

    print("    Rebuilding the same split and re-measuring...")
    print("(this re-extracts the features, so it takes a little while)")

    dataset = pd.read_csv(DATA_FILE)
    dataset = dataset.dropna(subset=["url", "label"])
    dataset["url"] = dataset["url"].astype(str).str.strip()
    dataset = dataset[dataset["label"].isin([0, 1])]
    dataset = dataset.drop_duplicates(subset=["url"], keep="first")
    dataset = dataset.reset_index(drop=True)
    dataset["is_phishing"] = (dataset["label"] == 0).astype(int)

    feature_table = features_to_dataframe(dataset["url"].tolist(),
                                          show_progress=True)
    features = feature_table.to_numpy()
    labels = dataset["is_phishing"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=labels,
    )

    model = bundle["model"]
    predictions = model.predict(x_test)

    recomputed = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    print()
    print(f"    {'Metric':<12}{'Stored at training':>22}"
          f"{'Recomputed now':>18}{'Same?':>9}")
    print("    " + "-" * 60)

    everything_matches = True
    for key, stored in bundle["metrics"].items():
        fresh = recomputed[key]
        matches = abs(stored - fresh) < 1e-9
        everything_matches = everything_matches and matches
        print(f"    {key:<12}{stored:>22.6f}{fresh:>18.6f}"
              f"{'YES' if matches else 'NO':>9}")

    print()
    if everything_matches:
        print("    PASS - the saved model reproduces the training results")
        print("           exactly. The reported numbers are genuine.")
    else:
        print("    WARNING - the numbers do not match. This can happen if")
        print("               data/dataset.csv was changed after training.")
        print("               Retrain with:  python train_model.py")


def predict_url(bundle, url):
    """Run one URL through the model. Returns a result dictionary."""
    model = bundle["model"]
    features = extract_features(url)
    vector = np.array(features_to_vector(features)).reshape(1, -1)

    prediction = int(model.predict(vector)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        # We remapped the data so class 1 always means PHISHING.
        confidence = float(probabilities[1])

    reasons = explain_features(features, bundle.get("importances"), top=5,
                               prediction=prediction)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "features": features,
        "reasons": reasons,
    }


def test_example_urls(bundle, extra_urls=None):
    """The testing section: run example URLs and print a clear report."""
    urls = list(EXAMPLE_URLS)
    if extra_urls:
        urls.extend(extra_urls)

    print()
    print("=" * 70)
    print("STEP 3 : TESTING EXAMPLE URLS")
    print("=" * 70)
    print(f"    Model in use: {bundle['best_model_name']}")
    print()
    print("    REMINDER: these are demo strings. The model reads only the")
    print("    TEXT of a link. It never opens the website.")
    print()

    for url in urls:
        result = predict_url(bundle, url)
        label = CLASS_NAMES[result["prediction"]]

        print("    " + "-" * 66)
        print(f"    URL entered        : {url}")

        feature_text = ", ".join(
            f"{name}={result['features'][name]}"
            for name in ["url_length", "num_dots", "has_https",
                         "has_ip_address", "has_suspicious_word",
                         "is_shortened_url", "risky_tld"]
        )
        print(f"    A few features     : {feature_text}")

        if result["confidence"] is not None:
            print(f"    Confidence (phishing) : "
                  f"{result['confidence'] * 100:.2f}%")
            print(f"    Confidence (legitimate): "
                  f"{(1 - result['confidence']) * 100:.2f}%")

        # The two lines that must never be confused with each other.
        print()
        print(f"    >> MODEL PREDICTION      : {label.upper()}")
        print(f"    >> CONFIRMED WEBSITE SAFETY: NOT ESTABLISHED")
        print(f"       (the model gives an opinion from the URL text only;"
              f" it is not proof)")

        if result["reasons"]:
            print()
            print(f"    Why the model leaned towards {label.upper()}:")
            for reason in result["reasons"]:
                print(f"       - {reason['feature']} = {reason['value']}"
                      f"   (model importance {reason['importance']})")
        print()

    print("=" * 70)
    print("END OF EVALUATION")
    print("=" * 70)
    print(HONESTY_NOTE)
    print("Next step:   streamlit run app.py")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Load the saved model, verify it and test example URLs."
    )
    parser.add_argument(
        "--url", action="append", dest="urls", default=[],
        help="Test an extra URL. Can be used more than once.",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the reproducibility check (it is the slow part).",
    )
    arguments = parser.parse_args()

    print()
    print("#" * 70)
    print("#  PhishGuard - Model Evaluation")
    print("#" * 70)

    bundle = load_bundle()

    if not arguments.skip_verify:
        verify_metrics(bundle)
    else:
        print()
        print("    (reproducibility check skipped)")

    test_example_urls(bundle, arguments.urls)


if __name__ == "__main__":
    main()
