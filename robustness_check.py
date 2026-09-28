"""
robustness_check.py
===================
WHY THIS FILE EXISTS
--------------------
A high accuracy on a test set does NOT prove a model works.

It only proves the model works on OTHER ROWS DRAWN FROM THE SAME DATASET.
If the dataset itself is skewed, the model can score 99% and still be
useless in the real world.

This project found exactly that, and this file is the evidence.

THE PROBLEM WE FOUND
--------------------
We checked one simple thing: does every legitimate URL in the PhiUSIIL
dataset start with a subdomain such as "www."?

    legitimate URLs with num_subdomains == 0 :    0  out of  134,850
    legitimate URLs with num_subdomains == 1 : 114,861 out of  134,850

Not one legitimate URL in the whole dataset looks like "https://abc.com".
They ALL look like "https://www.abc.com".

So a model trained on this data learns the rule:

        "no 'www.'  ->  phishing"

That rule is an artefact of HOW THE DATASET WAS COLLECTED, not a real
phishing behaviour. Try it on a real site and the model calls ordinary
websites dangerous. A 99% accuracy score hid this completely.

THE SECOND PROBLEM
------------------
All 134,850 legitimate URLs also have an EMPTY path - they are all bare
homepages such as "https://www.abc.com" with nothing after the domain.
So the model also learns "any path means phishing", which is again an
artefact rather than a real signal.

WHAT THIS SCRIPT DOES
---------------------
It builds a set of obviously-legitimate URLs by hand, checks them against
the trained model, and reports how many were wrongly flagged.

These URLs are NOT taken from the dataset. They were written by us to match
the SHAPE of a normal website. That is the whole point: the model has never
seen them, so if it gets them wrong we know it learned the wrong rule.

    python robustness_check.py

A LOW number in "wrongly flagged" is GOOD.
A HIGH number in "missed" is GOOD.

READING THE OUTPUT
------------------
"wrongly flagged"  = legitimate URLs called phishing.  BAD.
"caught"           = phishing URLs correctly called phishing.  GOOD.
"missed"           = phishing URLs let through.  BAD.
"""

import os

import joblib
import numpy as np
import pandas as pd

from feature_extraction import features_to_dataframe

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(PROJECT_FOLDER, "models", "phishguard_model.joblib")
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "dataset.csv")

# ---------------------------------------------------------------------------
# Our own test URLs. These are written by hand and are NOT in the dataset.
# Every one of them is a normal-looking website address.
# ---------------------------------------------------------------------------
NORMAL_BARE_DOMAINS = [
    "https://www.wikipedia.org/",
    "https://www.github.com/",
    "https://www.google.com/",
    "https://www.amazon.com/",
    "https://www.microsoft.com/",
    "https://stackoverflow.com/",
    "https://www.apple.com/",
    "https://www.netflix.com/",
    "https://www.coursera.org/",
    "https://www.ibm.com/",
]

NORMAL_NO_WWW = [
    "https://wikipedia.org/",
    "https://github.com/",
    "https://google.com/",
]

NORMAL_DEEP_LINKS = [
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://www.wikipedia.org/wiki/Phishing",
    "https://stackoverflow.com/questions/tagged/python",
]

# Obviously fake, used only to check the model still catches real attacks.
CLEAR_PHISHING = [
    "http://192.168.0.1/login",
    "http://secure-login.paypal-verify-account.xyz/login?session=9981",
    "http://my-bank.account-verify.tk/update.php?user=1024",
    "http://xn--pypal-4ve.com/signin",
    "https://bit.ly/3xK9abc",
    "http://paypal.com.secure-login.xyz/verify/account",
]


def show_dataset_skew(dataset):
    """Prove the claim that the legitimate class is unusually narrow."""
    from feature_extraction import features_to_dataframe

    print()
    print("=" * 70)
    print("STEP 1 : IS THE DATASET ITSELF SKEWED?")
    print("=" * 70)

    legitimate = dataset[dataset["label"] == 1]["url"].tolist()
    table = features_to_dataframe(legitimate)

    print()
    print("    How many LEGITIMATE urls have no subdomain (no 'www.')?")
    count = int((table["num_subdomains"] == 0).sum())
    print(f"        {count:,} out of {len(legitimate):,}")
    print()
    print("    How many LEGITIMATE urls have a non-empty path?")
    count_path = int((table["path_length"] > 0).sum())
    print(f"        {count_path:,} out of {len(legitimate):,}")
    print()
    print("    Breakdown of subdomain count in the legitimate class:")
    for value, times in table["num_subdomains"].value_counts().items():
        share = times / len(legitimate) * 100
        print(f"        {int(value)} subdomain(s) : {times:>8,}  ({share:5.2f}%)")
    print()
    print("    If the first two numbers are 0, then EVERY legitimate url in")
    print("    this dataset is a bare 'https://www.something.tld' homepage.")
    print("    A model can score very highly and still be useless in the")
    print("    real world, because it learned the collection method rather")
    print("    than how phishing works.")


def test_group(model, title, urls, expect_phishing):
    """Run a group of URLs and report how many were got right."""
    predictions = model.predict_proba(
        features_to_dataframe(urls).to_numpy()
    )[:, 1]
    flagged = predictions > 0.5

    if expect_phishing:
        correct = int(flagged.sum())
        print(f"    caught            : {correct}/{len(urls)}   "
              f"(higher is better)")
    else:
        wrong = int(flagged.sum())
        print(f"    wrongly flagged   : {wrong}/{len(urls)}   "
              f"(lower is better)")

    return flagged, predictions


def main():
    print()
    print("#" * 70)
    print("#  PhishGuard - Generalisation / Robustness Check")
    print("#" * 70)

    if not os.path.exists(MODEL_FILE):
        print()
        print("ERROR: no trained model found.")
        print("Run this first:  python train_model.py")
        raise SystemExit(1)

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    class_names = bundle.get("class_names", ["Legitimate", "Phishing"])

    print()
    print(f"    Model under test : {bundle['best_model_name']}")
    print(f"    Test F1 on the dataset's own test set : {bundle['metrics']['f1']:.4f}")
    print()
    print("    The question this script asks:")
    print("    The model scores well on the dataset. Does that mean it")
    print("    works on ordinary websites it has never seen?")

    # ---- step 1: is the dataset skewed? --------------------------------
    if os.path.exists(DATA_FILE):
        dataset = pd.read_csv(DATA_FILE)
        show_dataset_skew(dataset)
    else:
        print()
        print("    (data/dataset.csv not found - skipping the skew report)")

    # ---- step 2: test our own URLs -------------------------------------
    print()
    print("=" * 70)
    print("STEP 2 : TESTING URLs THE MODEL HAS NEVER SEEN")
    print("=" * 70)
    print("    These were written by hand. They are not in the dataset.")

    groups = [
        ("Normal sites, written the usual way (https://www.site.tld)",
         NORMAL_BARE_DOMAINS, False),
        ("Normal sites typed WITHOUT 'www.' (https://site.tld)",
         NORMAL_NO_WWW, False),
        ("Normal sites with a page in the address (/wiki/Phishing)",
         NORMAL_DEEP_LINKS, False),
        ("Obviously fake phishing links", CLEAR_PHISHING, True),
    ]

    for title, urls, expect_phishing in groups:
        print()
        print(f"  {title}")
        print("  " + "-" * 66)
        flagged, probabilities = test_group(model, title, urls, expect_phishing)
        for url, was_flagged, chance in zip(urls, flagged, probabilities):
            mark = "PHISHING" if was_flagged else "legitimate"
            print(f"      {mark:<11} {chance * 100:6.2f}%  {url}")

    # ---- step 3 : the honest conclusion --------------------------------
    print()
    print("=" * 70)
    print("WHAT THIS TELLS US")
    print("=" * 70)
    print("""
    The model scores about 0.96 F1 on the dataset's own test set, and that
    number is real. But the test rows come from the same collection process
    as the training rows, so a high score only proves the model fits THAT
    distribution.

    In the PhiUSIIL dataset every legitimate URL is a bare "https://www."
    homepage. A model therefore learns two shortcuts that are artefacts of
    data collection rather than real phishing behaviour:

        1. no "www." in the address  ->  phishing
        2. anything after the domain ->  phishing

    On real websites those two rules cause false alarms. This is a genuine
    limitation of the DATASET, not a coding mistake, and it cannot be fixed
    by inventing extra training data - doing that would be fabricating
    results.

    The honest conclusions for the report are:
      * report the test-set metrics as "performance on this dataset"
      * state clearly that real-world accuracy is expected to be LOWER
      * name a dataset with deeper legitimate URLs as future work
""")


if __name__ == "__main__":
    main()

