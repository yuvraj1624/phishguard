# PhishGuard - AI-Based Phishing Website Detection Using Machine Learning

A beginner-friendly BTech project that takes a website URL, works out **18
simple numbers** from it, and uses a trained machine learning model to guess
whether the link is **Legitimate** or **Phishing**.

Built with Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
and Joblib.

---

## Quick start (2 commands)

The trained model is already included in this repository, so you do **not**
need to download the dataset or retrain anything. Clone and run:

```powershell
git clone https://github.com/yuvraj1624/phishguard.git
cd phishguard
pip install -r requirements.txt
streamlit run app.py
```

Then open **<http://localhost:8501>**, type a URL, and press **Check URL**.

| You will see | Meaning |
|---|---|
| Accuracy 0.9673 · Precision 0.9821 · Recall 0.9405 · F1 0.9608 | Measured on a 20% test set the model never saw |

> ⚠️ **Read this before trusting the score.** All 134,850 legitimate URLs in
> the training dataset are bare `https://www.something.tld` homepages, so the
> model wrongly flags addresses typed without `www.` and any deep link. This is
> a property of the dataset, not a bug, and it is measured and explained in
> section 11.5. Reproduce it yourself with `python robustness_check.py`.

<details>
<summary>Optional: retrain the model from scratch</summary>

```powershell
python download_dataset.py     # fetches the real 235,795-row UCI dataset
python train_model.py          # trains 3 models, selects by F1, saves, plots
python evaluate_model.py       # proves the results reproduce exactly
```

</details>

---

## 1. Project Overview

Phishing is when a criminal builds a fake website that looks exactly like a
real one, so that you type your password into it. The fake address is usually
built in a hurry, and that rush leaves small traces **inside the link itself**:
it is very long, it has strange symbols, it shouts words like "verify" or
"secure", it hides behind a URL shortener, or it ends in a cheap domain like
`.xyz`.

PhishGuard reads those traces, turns them into numbers, and lets a machine
learning model judge the link.

**Important:** this is a first-check tool, not a security guarantee. See
[Limitations](#14-limitations) and especially
[the important finding](#115-the-most-important-finding--read-this-before-you-trust-the-score).

---

## 2. Problem Statement

Users are taught to "look at the address bar", but in practice a long,
confusing address is hard to judge while you are in a hurry. Existing
browser warnings only trigger on known-bad links, so brand-new phishing
sites slip through.

This project builds a small, explainable classifier that can give an instant
second opinion about a link **before** the user visits it, using only the text
of the URL — no downloading of the page, no DNS lookups, no WHOIS queries.
That keeps it fast, private and easy to explain.

---

## 3. Features

- 18 hand-crafted, easy-to-explain URL features
- Three models trained and compared: Logistic Regression, Decision Tree,
  Random Forest
- Automatic model selection using **F1-score on the phishing class**
- A clean Streamlit web interface
- Shows the prediction, a confidence score, the extracted features, and
  *why* the model decided that way
- 12 graphs and 4 report files generated automatically from real results
- An independent `robustness_check.py` that probes the model on URLs it has
  never seen, so its real limits are measured and not assumed
- 100% reproducible (fixed `random_state=42`)

---

## 4. Technologies Used

| Tool | Why we use it |
|---|---|
| **Python** | The language everything is written in |
| **Pandas** | Reading the CSV dataset and holding tables |
| **NumPy** | Fast maths on the arrays of numbers |
| **Scikit-learn** | The three machine learning models and all the metrics |
| **Matplotlib** | Drawing the graphs |
| **Seaborn** | Drawing the confusion matrix heatmaps |
| **Streamlit** | Building the web interface from Python alone |
| **Joblib** | Saving the trained model to a file and loading it back |

No React, no Node.js, no Docker, no cloud, no database.

---

## 5. Dataset

### Which dataset?

**PhiUSIIL Phishing URL (Website)** — from the UCI Machine Learning Repository.

- **Website:** <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- **Direct CSV link:** <https://archive.ics.uci.edu/static/public/967/data.csv>
- **Size:** about 54 MB, 235,795 rows, 56 columns
- **Licence:** CC BY 4.0
- **Reference paper:**
  Prasad, A., & Chandra, S. (2023). *PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning.* Computers & Security.
  DOI: [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)

### What the labels mean

The original dataset uses:

| Label | Meaning |
|---|---|
| `1` | Legitimate website |
| `0` | Phishing website |

`download_dataset.py` keeps that meaning and adds a readable `label_text`
column. **Inside training we flip it**, so that PHISHING becomes class `1`.
That is because every formula we use treats class `1` as the class we care
about catching, so "recall" then means *"out of all the real phishing sites,
how many did we catch?"*

### Getting the data

You do **not** need to download anything by hand. Just run:

```powershell
python download_dataset.py
```

It downloads the 54 MB file, keeps only the `url` and `label` columns, removes
broken and duplicate rows, writes a small clean `data/dataset.csv`, and then
deletes the big file to save space.

### If the automatic download fails

1. Open <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
2. Download the CSV
3. Rename it to `_raw_download.csv`
4. Put it in the `data\` folder
5. Run `python download_dataset.py` again

---

## 6. The 18 Features

| # | Feature | What it means |
|---|---|---|
| 1 | `url_length` | Total characters in the URL |
| 2 | `num_dots` | How many `.` characters |
| 3 | `num_hyphens` | How many `-` characters |
| 4 | `num_underscores` | How many `_` characters |
| 5 | `num_special_chars` | Characters that are not letters, digits, `.`, `-`, `_` or `/` |
| 6 | `num_digits` | How many numbers `0-9` |
| 7 | `num_subdirectories` | Folder levels, e.g. `/a/b/c` = 3 |
| 8 | `path_length` | Characters after the domain name |
| 9 | `domain_length` | Characters in the domain name |
| 10 | `num_query_params` | How many `?key=value` pairs |
| 11 | `has_https` | 1 if it starts with `https://` |
| 12 | `has_ip_address` | 1 if the site uses a raw IP instead of a name |
| 13 | `has_at_symbol` | 1 if the URL contains `@` |
| 14 | `has_punycode` | 1 if the domain contains `xn--` |
| 15 | `num_subdomains` | Levels before the main name |
| 16 | `has_suspicious_word` | 1 if a word like `login` / `verify` / `secure` appears |
| 17 | `is_shortened_url` | 1 if a known shortener (`bit.ly`, `tinyurl`...) is used |
| 18 | `risky_tld` | 1 if the ending is risky (`.xyz`, `.zip`, `.tk`...) |

Features 16, 17 and 18 use **fixed word lists written by hand from general
knowledge**. They were not learned or tuned from the data, so they are not
data leakage.

---

## 7. Installation (Windows)

Open **PowerShell** and run these commands one at a time.

```powershell
cd "C:\Users\yuvra\OneDrive\Documents\open code\phishguard"

# 1. Create an isolated environment for the project
python -m venv .venv

# 2. Switch into it
.\.venv\Scripts\Activate.ps1

# 3. Make pip up to date
python -m pip install --upgrade pip

# 4. Install everything the project needs
pip install -r requirements.txt
```

### If step 2 gives a red "running scripts is disabled" error

PowerShell blocks activation scripts by default. Run this once, then try
again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You will now see `(.venv)` at the start of the prompt. That means it worked.

### If you would rather not activate anything

Skip step 2 and 3 and use the full path to Python instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then replace `python` with `.\.venv\Scripts\python.exe` in every command below.

### Note on `py` vs `python`

Some tutorials use `py -3`. If your machine does not have the Python launcher
installed you will get *"py is not recognized"*. Just use `python` instead.

---

## 8. How to train the model

```powershell
python train_model.py
```

This takes a few minutes. It will:

1. Load and clean the dataset
2. Turn all 235,370 URLs into 18 numbers each
3. Split 80% train / 20% test with `random_state=42`
4. Train all three models
5. Print accuracy, precision, recall, F1 and the confusion matrix for each
6. Pick the model with the best **F1 on the phishing class**
7. Save it to `models\phishguard_model.joblib`
8. Draw 12 graphs into `results\`
9. Write 4 report files into `results\`

### Faster run, for testing only

```powershell
python train_model.py --sample 20000
```

This trains on a 20,000-row sample. Useful for checking everything works, but
**do not use these numbers in your report** — run the full version for that.

---

## 9. How to evaluate the model

```powershell
python evaluate_model.py
```

This loads the saved model file, then:

- re-runs the identical test to prove the numbers are reproducible
- runs a set of example URLs and prints a result for each

To test your own URLs:

```powershell
python evaluate_model.py --url "https://www.example.com"
```

To skip the slow reproducibility check:

```powershell
python evaluate_model.py --skip-verify
```

---

## 10. How to start the web application

```powershell
streamlit run app.py
```

Streamlit prints a local address, usually <http://localhost:8501>. Open it in
your browser. Type a URL, press **Check URL**, and read the result.

Stop the app with `Ctrl+C` in the PowerShell window.

---

## 11. Results (measured, not estimated)

Every number below was produced by running this code against the real dataset.
Nothing here is copied from a paper or written by hand.

### 11.1 Dataset after cleaning

| Class | Rows | Share |
|---|---|---|
| Legitimate | 134,850 | 57.29% |
| Phishing | 100,520 | 42.71% |
| **Total** | **235,370** | 100% |

Split 80 / 20 into 188,296 training rows and 47,074 test rows, with
`random_state=42` and `stratify`, so the split is identical every run.

### 11.2 Model comparison on the held-out test set

| Model | Accuracy | Precision | Recall | F1 | Selected |
|---|---|---|---|---|---|
| Logistic Regression | 0.9526 | 0.9772 | 0.9104 | 0.9426 | no |
| Decision Tree | 0.9667 | 0.9822 | 0.9391 | 0.9602 | no |
| **Random Forest** | **0.9673** | **0.9821** | **0.9405** | **0.9608** | **YES** |

**The Random Forest is selected automatically because it has the highest F1 on
the phishing class.** The Decision Tree is only 0.0006 behind, so either would
have been a defensible choice — we report that honestly rather than
overclaiming.

### 11.3 Confusion matrix of the selected model

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actually Legitimate** | 26,625 (correct) | 345 (false alarm) |
| **Actually Phishing** | 1,196 (**missed**) | 18,908 (caught) |

In plain words: of every 100 phishing links, about 94 were caught. About 1 in
17 got through. Only 1.28% of legitimate links were false alarms.

### 11.4 Graphs

All 12 graphs are written to `results\` by `train_model.py`:

`class_distribution.png`, `accuracy_comparison.png`,
`precision_comparison.png`, `recall_comparison.png`, `f1_comparison.png`,
`all_metrics_comparison.png`, `confusion_matrix_logistic_regression.png`,
`confusion_matrix_decision_tree.png`, `confusion_matrix_random_forest.png`,
`confusion_matrix_best_model.png`, `feature_importance.png`,
`predicted_vs_actual.png`.

---

### 11.5 The most important finding — read this before you trust the score

**A 96% accuracy does not automatically mean a working detector.** We tested
the trained model against ordinary website addresses we wrote by hand, which
it had never seen:

| Test group | Result |
|---|---|
| Normal sites, usual form (`https://www.site.tld`) | 1 of 10 wrongly flagged |
| Normal sites without `www.` (`https://site.tld`) | **3 of 3 wrongly flagged** |
| Normal sites with a page (`/wiki/Phishing`) | **3 of 3 wrongly flagged** |
| Obviously fake phishing links | **6 of 6 correctly caught** |

It still catches every fake link, but it wrongly calls addresses like
`https://google.com/` and `https://en.wikipedia.org/wiki/Phishing` phishing.

### Why this happens

We measured the dataset directly:

| Question about the 134,850 legitimate URLs | Answer |
|---|---|
| How many have no subdomain (no `www.`)? | **0** |
| How many have a non-empty path? | **0** |

**Every single legitimate URL in the dataset is a bare
`https://www.something.tld` homepage.** Not one exception in 134,850.

So a model can reach 96% by learning just two shortcuts:

1. **No `www.` → phishing**
2. **Anything after the domain → phishing**

Both fit the dataset perfectly. Neither describes how phishing really works.
This is a property of **how the dataset was collected**, not a bug in the code.

### Why the test set did not catch it

This is the key lesson. The test rows come from the same collection process as
the training rows, so a test score measures *"how well does the model fit this
dataset"* — not *"will this work on the internet"*. A test set drawn from a
biased distribution cannot reveal the bias in that distribution.

### We did not "fix" it by inventing data

The only way to make the model handle bare domains and deep links would be to
add legitimate examples we made up ourselves. That is fabricating data, and it
would make every number in the report meaningless. Instead the problem is
measured, disclosed in the app and in the report, and fixing it is the first
item under [Future Scope](#15-future-scope).

Reproduce all of this yourself:

```powershell
python robustness_check.py
```

> **A separate, real bug that we did fix:** during development, typing
> `https://www.wikipedia.org/` was classified as **phishing with 100%
> confidence** while `https://www.wikipedia.org` was correctly called
> legitimate. The dataset happens to contain one phishing row sitting at
> exactly `url_length = 26`, and the trailing slash pushed the real Wikipedia
> into that memorised spot. The project now rebuilds every URL into a single
> canonical form (`example.com` and `example.com/` are the same page) and
> measures **all** features from that, so the two spellings now produce
> identical numbers. Fixing this deliberately *lowered* the measured F1 from
> 0.9951 to 0.9608 — the old higher score was partly the model exploiting the
> "any path means phishing" artefact. A lower number that reflects a better
> system is worth more than a higher number that hides a bug.

---

## 12. Project Structure

```
phishguard/
│
├── app.py                     Streamlit web interface
├── train_model.py             Trains, compares, selects and saves the model
├── evaluate_model.py          Verifies the saved model and tests example URLs
├── robustness_check.py        Probes the model on URLs it has never seen
├── feature_extraction.py      Turns a URL into the 18 features
├── download_dataset.py        Downloads and cleans the real dataset
├── requirements.txt           The packages you need
├── README.md                  This file
├── PROJECT_EXPLANATION.md     Line-by-line guide + viva preparation
├── REPORT.md                  Research paper content with real results
│
├── data/
│   └── dataset.csv            Clean dataset (url, label, label_text)
│
├── models/
│   └── phishguard_model.joblib    The trained model + everything needed
│
└── results/
    ├── class_distribution.png
    ├── accuracy_comparison.png
    ├── precision_comparison.png
    ├── recall_comparison.png
    ├── f1_comparison.png
    ├── all_metrics_comparison.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_best_model.png
    ├── feature_importance.png
    ├── predicted_vs_actual.png
    ├── dataset_statistics.csv
    ├── model_comparison.csv
    ├── metrics_report.txt
    └── best_model.txt
```

### What each file does, in one line

| File | Job |
|---|---|
| `feature_extraction.py` | URL text → 18 numbers |
| `download_dataset.py` | Internet → `data\dataset.csv` |
| `train_model.py` | Dataset → trained, saved model + graphs |
| `evaluate_model.py` | Saved model → proof it works and is reproducible |
| `robustness_check.py` | Saved model → proof of where it breaks |
| `app.py` | Saved model → web page |

`feature_extraction.py` is the only file that knows *how* to read a URL. Both
training and the app call the same function, so they can never disagree.

---

## 13. Full Command Sequence (copy-paste)

```powershell
cd "C:\Users\yuvra\OneDrive\Documents\open code\phishguard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python train_model.py
python evaluate_model.py
python robustness_check.py
streamlit run app.py
```

---

## 14. Limitations

Be honest about these in your report — examiners like it.

1. **The dataset's legitimate class is severely skewed.** All 134,850
   legitimate URLs are bare `https://www.…` homepages, so the model learns
   "no `www.` means phishing" and "any path means phishing". This is the
   biggest limitation and it is measured in section 11.5.
2. **Real-world accuracy will be lower than 0.9673.** The test set shares the
   training set's bias, so it cannot reveal the failure modes.
3. **It only reads the link, never the website.** A perfectly ordinary-looking
   link can still lead to a harmful page, and a brand-new phishing site can
   look clean.
4. **"Legitimate" is not "safe".** The model predicts a *pattern*, not a fact.
   It has no knowledge of what is actually on the page.
5. **1 in 17 real phishing links was missed** on the test set (1,196 of
   20,104), and that figure is optimistic for the same reason as above.
6. **Zero-day phishing will be missed.** Attackers who know these 18 features
   could design a URL that deliberately scores as normal.
7. **A decision tree can be over-fitted.** A single tree can memorise the
   training data. Random Forest exists to fix this.
8. **Four features contribute nothing** (`has_at_symbol`, `has_punycode`,
   `is_shortened_url`, `has_ip_address`) because this dataset's URLs were
   normalised during collection. They detect real techniques, so this is a gap
   in the data rather than in the features.
9. **The dataset is not the live internet.** It was collected at one point in
   time. Attack trends change, so the model can slowly go out of date.
10. **The three keyword lists are hand-written.** They reflect general
    knowledge, not measured frequencies, and they will always be incomplete.
11. **No deep learning, no page content.** We deliberately avoid HTML scraping,
    DNS lookups and WHOIS queries to keep the project simple and fast.

---

## 15. Future Scope

- **Use a dataset whose legitimate class includes deep links and bare
  domains.** This single change would improve real-world accuracy more than
  any other item here, and it directly fixes the finding in section 11.5.
- Drop the redundant features (the model already gets the answer from
  `has_https` and `num_digits`) and check whether accuracy holds up
- Add more features (URL entropy, character ratios, TLD reputation lists)
- Try gradient boosting and compare against the current winner
- Fetch the real page and inspect forms, password fields and external links
- Add WHOIS age and DNS records as extra signals
- Add a browser extension that checks every link before you click
- Keep the model fresh with a scheduled retrain on new data
- Turn it into a REST API for other tools to call
- Extend it from URLs to full emails and SMS messages

---

## 16. Licence and Credit

Project code: written for academic use.

Dataset: PhiUSIIL Phishing URL (Website), UCI Machine Learning Repository,
CC BY 4.0. Please cite the original paper if you use it.
