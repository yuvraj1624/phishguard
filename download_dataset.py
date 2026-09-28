"""
download_dataset.py
===================
This script gets the REAL phishing dataset and turns it into a small,
clean file that the rest of the project can use.

WHERE DOES THE DATA COME FROM?
------------------------------
Dataset : PhiUSIIL Phishing URL (Website)
Source  : UCI Machine Learning Repository
Website : https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
Paper   : Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security
          profile empowered phishing URL detection framework based on
          similarity index and incremental learning. Computers & Security.
          DOI: https://doi.org/10.1016/j.cose.2023.103545
Licence : CC BY 4.0

The original file is about 54 MB and has 56 columns. Most of those columns
are things you can only get by actually DOWNLOADING and reading the web page
(for example "NumberOfImages", "HasFavicon", "LineOfCode"). We cannot get
those in real time when a user types a URL, so we throw them away and keep
only the two columns we need:

    url   - the web address
    label - 0 means PHISHING, 1 means LEGITIMATE

WHY NOT USE THE ORIGINAL FILE DIRECTLY?
---------------------------------------
Two reasons:
  1. It is 54 MB. Reading it every time is slow.
  2. Its "Title" column contains broken (non UTF-8) characters, which makes
     the file fail to load unless you ask for a special encoding.

This script downloads it once, keeps only what we need, and writes:

    data/dataset.csv      (about 20 MB, 3 columns, clean)

HOW TO RUN:
    python download_dataset.py
"""

import argparse
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
CLEAN_DATASET = os.path.join(DATA_FOLDER, "dataset.csv")
RAW_DOWNLOAD = os.path.join(DATA_FOLDER, "_raw_download.csv")

# The official UCI download link for this dataset.
DATASET_URL = ("https://archive.ics.uci.edu/static/public/967/"
               "phiusiil+phishing+url+dataset.zip")

# A second, simpler link that returns the CSV directly.
DATASET_URL_CSV = ("https://archive.ics.uci.edu/static/public/967/data.csv")

# How many rows we expect. Used only to warn you if the download is wrong.
EXPECTED_MINIMUM_ROWS = 200000


# ---------------------------------------------------------------------------
# Step 1 - download the original file
# ---------------------------------------------------------------------------
def download_raw_file():
    """
    Download the original UCI file into data/_raw_download.csv

    Returns True if it worked, False if it failed.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)

    print("=" * 70)
    print("STEP 1 of 2 : downloading the real dataset from UCI")
    print("=" * 70)
    print("Source :", DATASET_URL_CSV)
    print("This file is about 54 MB. Please wait...")

    request = urllib.request.Request(
        DATASET_URL_CSV,
        headers={"User-Agent": "Mozilla/5.0 (PhishGuard student project)"},
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_report = 0

            with open(RAW_DOWNLOAD, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)

                    # Print progress roughly every 10 MB
                    if downloaded - last_report >= 10 * 1024 * 1024:
                        if total_bytes:
                            percent = downloaded / total_bytes * 100
                            print(f"    {percent:5.1f}%  "
                                  f"({downloaded / 1048576:.0f} MB)")
                        else:
                            print(f"    {downloaded / 1048576:.0f} MB")
                        last_report = downloaded

        size_mb = downloaded / 1048576
        print()
        print(f"    Download finished. File size = {size_mb:.1f} MB")

        if size_mb < 1:
            print("    ERROR: the downloaded file is far too small.")
            return False

        return True

    except Exception as error:
        print()
        print("    DOWNLOAD FAILED:", error)
        print()
        print("    What to do instead:")
        print("    1. Open this page in your browser:")
        print("       https://archive.ics.uci.edu/dataset/967/"
              "phiusiil+phishing+url+dataset")
        print("    2. Download the CSV file.")
        print("    3. Rename it to  _raw_download.csv")
        print("    4. Put it inside the  data  folder.")
        print("    5. Run this script again.")
        return False


# ---------------------------------------------------------------------------
# Step 2 - clean it up
# ---------------------------------------------------------------------------
def build_clean_dataset():
    """
    Read the big original file, keep only url + label, remove bad rows,
    and write the small clean file data/dataset.csv
    """
    import pandas as pd

    print()
    print("=" * 70)
    print("STEP 2 of 2 : cleaning the data")
    print("=" * 70)

    # encoding='latin-1' is REQUIRED. The original file contains broken
    # characters in its "Title" column and will crash if we use utf-8.
    print("Reading the original file (this takes a moment)...")
    original = pd.read_csv(
        RAW_DOWNLOAD,
        encoding="latin-1",       # stops the broken characters causing an error
        low_memory=False,
        on_bad_lines="skip",      # quietly skip any row that is malformed
    )
    print(f"    Original file : {len(original):,} rows, "
          f"{original.shape[1]} columns")

    # Find the URL column and the label column, whatever their exact case is.
    url_column = None
    label_column = None

    for column in original.columns:
        name = str(column).strip().lower()
        if name == "url":
            url_column = column
        elif name == "label":
            label_column = column

    if url_column is None or label_column is None:
        print()
        print("    ERROR: could not find the 'URL' and 'label' columns.")
        print("    Columns found:", list(original.columns))
        return False

    print(f"    URL column   : '{url_column}'")
    print(f"    Label column : '{label_column}'")

    # Keep only the two columns we need.
    small = original[[url_column, label_column]].copy()
    small.columns = ["url", "label"]

    # --- cleaning step 1: remove rows with a missing URL -------------------
    rows_before = len(small)
    small = small.dropna(subset=["url", "label"])
    small["url"] = small["url"].astype(str).str.strip()
    small["label"] = pd.to_numeric(small["label"], errors="coerce")
    small = small[small["url"].str.len() > 0]
    small = small.dropna(subset=["label"])
    print(f"    Removed {rows_before - len(small):,} rows with a missing "
          f"URL or label")

    # --- cleaning step 2: remove rows whose label is not 0 or 1 ------------
    rows_before = len(small)
    small = small[small["label"].isin([0, 1])]
    small["label"] = small["label"].astype(int)
    print(f"    Removed {rows_before - len(small):,} rows with an "
          f"invalid label")

    # --- cleaning step 3: remove exact duplicate URLs ----------------------
    # This also protects us from data leakage, where the SAME url could
    # appear in both the training set and the test set.
    rows_before = len(small)
    small = small.drop_duplicates(subset=["url"], keep="first")
    print(f"    Removed {rows_before - len(small):,} duplicate URLs")

    # --- add a readable text column so the CSV is easy to understand -------
    small["label_text"] = small["label"].map({0: "phishing", 1: "legitimate"})

    small = small.reset_index(drop=True)

    # --- save ---------------------------------------------------------------
    small.to_csv(CLEAN_DATASET, index=False)

    print()
    print(f"    Clean file   : {CLEAN_DATASET}")
    print(f"    Final rows   : {len(small):,}")
    print(f"    File size    : {os.path.getsize(CLEAN_DATASET) / 1048576:.1f} MB")
    print()
    print("    Class balance (this is a REAL count taken from the file):")
    counts = small["label_text"].value_counts()
    for name, count in counts.items():
        share = count / len(small) * 100
        print(f"        {name:<12} {count:>8,}   ({share:5.2f}%)")

    return len(small) >= EXPECTED_MINIMUM_ROWS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("PhishGuard - dataset preparation")
    print("Project folder:", PROJECT_FOLDER)
    print()

    parser = argparse.ArgumentParser(
        description="Download and clean the PhiUSIIL phishing URL dataset."
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the big 54 MB original file instead of deleting it.",
    )
    arguments = parser.parse_args()

    # If a clean file is already there and the user did not ask for a
    # fresh download, we can reuse it.
    if os.path.exists(CLEAN_DATASET) and not os.path.exists(RAW_DOWNLOAD):
        answer = input(
            "\ndata/dataset.csv already exists. Download it again? (y/N): "
        )
        if answer.strip().lower() != "y":
            print("Keeping the existing data/dataset.csv. Nothing to do.")
            return

    if not os.path.exists(RAW_DOWNLOAD):
        if not download_raw_file():
            sys.exit(1)
    else:
        print("Using the original file already present at:",
              RAW_DOWNLOAD)

    success = build_clean_dataset()

    # Delete the big file so the project folder stays small.
    if os.path.exists(RAW_DOWNLOAD) and not arguments.keep_raw:
        size_mb = os.path.getsize(RAW_DOWNLOAD) / 1048576
        os.remove(RAW_DOWNLOAD)
        print()
        print(f"    Deleted the {size_mb:.0f} MB original file to save space.")
        print("    (run with --keep-raw if you want to keep it)")

    if not success:
        print()
        print("WARNING: the dataset looks smaller than expected.")
        print("Expected at least", EXPECTED_MINIMUM_ROWS, "rows.")
        print("Check the download and run this script again.")

    print()
    print("DONE. Next step:  python train_model.py")
    print()


if __name__ == "__main__":
    main()
