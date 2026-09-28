# PhishGuard - Project Explanation & Viva Preparation

This document explains, in plain language, how every file in the project
works and why it was written that way. It is meant to be read before a viva.

Every number quoted here was produced by running the code and can be
reproduced with the commands in the README.

---

## Part 1 - The big idea, in one paragraph

A phishing site is a fake website built to steal your details. Attackers build
these quickly, and that speed leaves marks on the web address itself.
PhishGuard reads those marks, turns the address into **18 numbers**, and hands
the numbers to a **Random Forest** classifier, which was trained on 235,370
real labelled URLs and returns either *Phishing* or *Legitimate*.

---

## Part 2 - The machine learning workflow, in simple words

Think of it as a student preparing for an exam using past papers.

1. **LOAD (collect past papers)**
   We open `data/dataset.csv`, which holds 235,370 real URLs with the
   correct answer already written down.

2. **CLEAN (throw away bad questions)**
   Rows with a missing URL or an impossible label are removed, and exact
   duplicate URLs are deleted. On this dataset nothing was broken and 425
   duplicate URLs were removed, leaving 235,370 rows.

3. **FEATURES (work out what to notice)**
   This is the most important creative step. We cannot feed raw text to a
   model, so we convert each URL into 18 numbers. Example:

   ```
   http://secure-login.paypal-verify-account.xyz/login?session=9981
   becomes
   url_length=64, num_dots=2, num_hyphens=3, num_digits=4,
   path_length=6, has_https=0, has_suspicious_word=1, risky_tld=1, ...
   ```

4. **LABELS (be clear about what "1" means)**
   The original dataset uses `1 = legitimate, 0 = phishing`. We **flip it**
   so that inside our code `1 = phishing`. This matters because every
   formula in Part 4 treats class 1 as the class we care about catching.
   So "recall" now means *"of all the real phishing links, how many did we
   catch?"* - the number a security tool actually cares about.

5. **SPLIT (keep the exam hidden)**
   80% of the rows go to *training*, 20% go to *test*. The model never sees
   the test rows while it learns. This is the whole point - it is how we
   know the model did not just memorise the answers.

6. **TRAIN (practise)**
   Fit three different models on the training rows.

7. **MEASURE (take the hidden exam)**
   Run each model once on the test rows and calculate accuracy, precision,
   recall, F1 and the confusion matrix.

8. **CHOOSE (pick the best student)**
   Keep the model with the highest F1-score on the phishing class.

9. **SAVE and DEPLOY**
   Save the winner to `models/phishguard_model.joblib`. The Streamlit app
   loads that file and never retrains anything.

---

## Part 3 - Why each file exists

### `feature_extraction.py` - the heart of the project

Everything else depends on this file, so read it carefully.

| Part of the file | What it does |
|---|---|
| `SUSPICIOUS_WORDS` | 33 hand-written words: `login`, `verify`, `secure`, `wallet`... |
| `URL_SHORTENERS` | 30 services that hide the real destination: `bit.ly`, `tinyurl.com`... |
| `RISKY_TLDS` | 37 cheap or spammy domain endings: `.xyz`, `.zip`, `.tk`... |
| `FEATURE_NAMES` | The list of 18 feature names, **in a fixed order**. This order is the contract between training and prediction. |
| `FEATURE_DESCRIPTIONS` | A plain-English meaning for each feature, used by the app's table |
| `FEATURE_DIRECTION` | Tells the app whether a big value means "more suspicious" or "more normal". Used **only for explanation**, never for training. |
| `normalise_url()` | Cleans what the user typed. If they type `google.com` it becomes `http://google.com` so the features still make sense. |
| `_canonical_path()` | Removes a lone trailing `/`, because `example.com` and `example.com/` are the same page. |
| `_rebuild_canonical_url()` | Rebuilds the whole address in one standard form, and **every** feature is measured from that. |
| `extract_features()` | **The main function.** URL string in, dictionary of 18 numbers out. |
| `features_to_vector()` | Turns the dictionary into a plain list, in `FEATURE_NAMES` order, because scikit-learn wants a list. |
| `features_to_dataframe()` | Runs `extract_features()` over the whole dataset. |
| `explain_features()` | Builds the "why did it say that?" list for the app. |

**Decision 1 - one source of truth**

Both `train_model.py` and `app.py` import the *same* `extract_features()`
function. The model is therefore trained on exactly the same 18 numbers that
it will be asked to judge in the app.

A very common mistake in student projects is to train on features that came
with the dataset, but then compute *different* features at prediction time.
The model then receives numbers it has never seen and produces nonsense. We
avoid that by having one single source of truth.

**Decision 2 - why we only look at the URL text, not the website**

The original dataset has 54 extra columns such as *HasFavicon*,
*NumberOfImages* and *LineOfCode*. Those require actually downloading the web
page. We cannot do that when a user types a URL into our app, so we discard
them. Using only the URL text keeps the app instant, private and simple.

**Decision 3 - URL canonicalisation (this one fixed a real bug)**

During testing, typing `https://www.wikipedia.org/` returned **PHISHING with
100% confidence**, while `https://www.wikipedia.org` was correctly called
legitimate. The dataset happens to contain a single phishing row sitting at
exactly `url_length = 26`, and the trailing slash pushed the real Wikipedia
into that one memorised spot. The Decision Tree had 755 leaves and had drawn a
sharp boundary around a single point.

`https://example.com` and `https://example.com/` are the same page, so we now
rebuild every URL into one standard form and measure **all** features from
that. The two spellings now produce identical numbers.

Interestingly this **lowered** the measured F1 from 0.9951 to 0.9608. That is
the *honest* result: the old higher score was partly the model exploiting the
artefact that "any URL with a path is phishing". Removing a shortcut makes the
score worse and the system better.

**About the three hand-written word lists - is this cheating?**

No. Here is the argument if an examiner asks:

- The lists were written from **general human knowledge** about how phishing
  works, before looking at any results.
- They were **not** tuned, edited or filtered using the training or test data.
- The model still has to *learn by itself* how much each of those numbers is
  worth. We never told it the answer.

If we had, for example, picked the suspicious words by looking at which words
appeared most often in phishing rows of the **test** set, that *would* be data
leakage. We did not do that.

### `download_dataset.py`

- Downloads the real 54 MB CSV from the UCI repository.
- Reads it with `encoding='latin-1'`. **This is not optional.** The original
  file contains broken non-UTF-8 bytes in its `Title` column, and reading it
  with the default UTF-8 encoding crashes.
- Keeps only the `URL` and `label` columns.
- Removes missing values, impossible labels and duplicate URLs.
- Adds a `label_text` column so the CSV is readable by a human.
- Deletes the 54 MB original so the project folder stays small.
- If the download fails, it prints step-by-step manual instructions.

### `train_model.py`

The 10 numbered steps at the top of the file match Part 2 exactly. The most
important parts:

- **`load_dataset()`** does steps 1, 2 and 4, including the label flip
  `is_phishing = (label == 0)`.
- **`build_feature_table()`** does step 3.
- **`split_data()`** does step 5. It uses `random_state=42` and
  `stratify=labels`, so the split is identical on every run and both halves
  keep the same class balance.
- **`build_models()`** creates the three models.
- **`measure()`** does step 7 for one model.
- **`get_importances()`** asks the model which features it leaned on most.
  Tree models give `feature_importances_`; Logistic Regression gives
  `coef_`, so we take the absolute value of those weights.
- **Model selection** is one line: `max(all_metrics, key=lambda n: ...['f1'])`.
- **`save_model()`** writes a single `.joblib` file containing the fitted
  model, the feature names, the importances and the measured metrics. Saving
  them together means the app can never be out of step with the model.

**Why Logistic Regression uses a Pipeline**

Logistic Regression is distance-based, so the features must be on a similar
scale. A URL of length 60 would otherwise drown out a 0/1 flag.
`StandardScaler` fixes that.

Crucially, the scaler is placed **inside** a `Pipeline`:

```python
Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(...))])
```

Scikit-learn then fits the scaler on the training rows only, every single
time. If we had called `scaler.fit_transform(x_train)` ourselves and kept the
scaler loose, it would be very easy to accidentally fit it on the test data
too. That is a classic data-leakage bug, and the Pipeline makes it impossible.

Decision Tree and Random Forest do not need scaling, so they are used raw.
Tree models split on thresholds like "is length > 40?", so scale does not
matter.

**Why no hyperparameter tuning?**

We set only `random_state` and left everything else at the default. Tuning
hyperparameters against the test set is itself a form of leakage, and for a
project of this scope the loss of a clean claim about generalisation is not
worth a small accuracy gain. This is a good answer if an examiner asks why you
did not "improve" the model.

### `evaluate_model.py`

Training creates the model; this file **checks** it. It has three jobs:

1. `load_bundle()` - load the `.joblib` file, which proves it is not corrupt.
2. `verify_metrics()` - rebuild the identical split and re-measure. If the
   numbers match the stored ones exactly, the results are proven reproducible.
   Our run matched to six decimal places on all four metrics.
3. `test_example_urls()` - run demo URLs and print two clearly separated lines:
   ```
   >> MODEL PREDICTION        : PHISHING
   >> CONFIRMED WEBSITE SAFETY: NOT ESTABLISHED
   ```
   That wording is deliberate and should be preserved. The example list
   deliberately includes two legitimate websites the model gets **wrong**, so
   the limitation is visible instead of hidden.

### `app.py`

A standard Streamlit script. It loads the model with
`@st.cache_resource` so the file is read from disk only once, then:

- shows the input box and the **Check URL** button
- calls `extract_features()` - the same function training used
- calls `model.predict()`
- shows the verdict, a confidence bar, verdict-aligned reasons, and a
  collapsible table of all 18 features
- **always** shows the disclaimer, whether the answer was good or bad
- has an expander explaining the known limitation

### `robustness_check.py`

This file exists because of a genuine finding, explained in Part 7.

---

## Part 4 - How to explain the four metrics

Imagine a **spam filter** for email. We predict the class `Phishing`, so
everything below is about phishing.

### Confusion matrix

A table of real answers against guessed answers.

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actually Legitimate** | **TN** = 26,625 | **FP** = 345 (false alarm) |
| **Actually Phishing** | **FN** = 1,196 (MISSED) | **TP** = 18,908 |

*(these are the real numbers for the selected Random Forest)*

- **TP (True Positive)** - a real phishing link, correctly caught.
- **TN (True Negative)** - a real legitimate link, correctly allowed.
- **FP (False Positive)** - a good link wrongly blocked. Annoying, but not dangerous.
- **FN (False Negative)** - a phishing link that got through. **This is the
  dangerous one.** A user could be scammed because of it.

### Accuracy
*"Out of every 100 links, how many did we classify correctly?"*

```
Accuracy = (TP + TN) / everything
         = (18,908 + 26,625) / 47,074
         = 0.9673   ->  96.73%
```

**Weakness:** accuracy lies when one class is much bigger than the other. A
model that always answered "Legitimate" would score 57.29% here, which sounds
poor - but it would also miss **every single phishing link**. This is why we
never rely on accuracy alone.

### Precision
*"When we say a link is phishing, how often are we right?"*

```
Precision = TP / (TP + FP)
          = 18,908 / (18,908 + 345)
          = 0.9821   ->  98.21%
```

**In plain words:** out of every 100 links we flagged, about 98 really were
phishing. Only 345 out of 26,970 legitimate links - about **1.28%** - were
false alarms.

Low precision would mean we block far too many honest websites.

### Recall (Sensitivity)
*"Out of all the phishing links that exist, how many did we catch?"*

```
Recall = TP / (TP + FN)
       = 18,908 / (18,908 + 1,196)
       = 0.9405   ->  94.05%
```

**In plain words:** we caught 94 out of every 100 real phishing links. About
**1 in 17** phishing links still got through. And remember - this figure is
measured on dataset-like data, so on real fresh attacks it will be worse.

Low recall would mean attackers slip past us. For a security tool this is the
more embarrassing error, so we care about it a lot.

### F1-score
*"Give me one number that punishes both mistakes at once."*

F1 is the **harmonic mean** of precision and recall. A harmonic mean sits
close to the *smaller* of the two values, so it cannot be high unless *both*
are high.

```
F1 = 2 x (Precision x Recall) / (Precision + Recall)
   = 2 x (0.9821 x 0.9405) / (0.9821 + 0.9405)
   = 0.9608   ->  96.08%
```

**Why we chose F1 for model selection:** a phishing filter has two opposite
failure modes. Too strict and it misses real attacks (low recall). Too
careless and it blocks honest users (low precision). F1 punishes both, so
maximising F1 forces a fair balance. Using accuracy alone would not have
penalised the model that misses phishing links.

### Our measured results

| Model | Accuracy | Precision | Recall | F1 | Training time |
|---|---|---|---|---|---|
| Logistic Regression | 0.9526 | 0.9772 | 0.9104 | 0.9426 | 0.29 s |
| Decision Tree | 0.9667 | 0.9822 | 0.9391 | 0.9602 | 0.49 s |
| **Random Forest (selected)** | **0.9673** | **0.9821** | **0.9405** | **0.9608** | 3.99 s |

---

## Part 5 - Why we chose these three models

### Logistic Regression - the simple baseline

It draws one straight line through the data and decides which side each URL
falls on. We chose it because it is the simplest possible classifier, it
trains almost instantly, and it gives a real probability rather than just
yes/no. Every good project needs a baseline to compare against - if a
complicated model cannot beat a straight line, it is not earning its keep.

**Result:** the weakest of the three (F1 = 0.9426). A single straight line
cannot capture the way several weak signals combine into a strong one. That
is exactly the limitation we expected, and it is a good thing to report.

### Decision Tree - the explainable one

It asks questions one after another: *"Is the URL longer than 40 characters?
No. Does it use a risky domain ending? Yes. Is it shorter than 15 characters?
Then it is phishing."* We chose it because it is easy to draw and easy to
explain to a non-technical person.

**Result:** very strong (F1 = 0.9602), and it won on precision (0.9822). But
a single tree grows to fit the training data very precisely, which risks
**overfitting**: memorising the training set instead of learning a general
pattern. We saw this directly - it drew a boundary around one memorised row
that flagged the real Wikipedia as phishing.

### Random Forest - the winner

It grows 100 decision trees, but each one sees a different random sample of
the data and a different random subset of features at every split. They all
vote, and the majority wins. We chose it because the "many independent trees
voting" trick is the standard cure for the overfitting that ruins a single
tree, and because voting also smooths out the noise in any one tree.

**Result:** the best F1 (0.9608), so it was selected.

**An honest note on how close it was.** The Decision Tree and Random Forest
are separated by only 0.0006 in F1. The tree actually has marginally *higher*
precision, and the forest marginally *higher* recall. On this test set either
would have been a defensible choice, and we should say so rather than
overclaiming. The forest was selected by the stated rule, and it is also the
safer choice on inputs unlike the training data.

**Why the very small gap is itself a finding.** Both tree models score around
0.96 while Logistic Regression scores 0.94. The gap between the simple
baseline and the complex models is narrow, which suggests the dataset's
structure is easy to exploit - and Part 7 shows that what the trees are
exploiting is partly an artefact.

---

## Part 6 - Avoiding data leakage

"Leakage" means letting the model accidentally see information it should not,
which makes results look better than they really are. We blocked it four ways:

1. **Split first, fit later.** `train_test_split` happens before any model or
   scaler is fitted. The test rows are untouched until the single measuring step.
2. **Scaling inside a Pipeline.** `StandardScaler` learns its averages from
   the training rows only, automatically, every time.
3. **Duplicate URLs removed.** If the identical URL appeared in both train and
   test, the model would be tested on something it had memorised. 425
   duplicates were removed in the cleaning step.
4. **Fixed `random_state=42`.** The split, the tree and the forest are all
   seeded, so the numbers in the report can be reproduced exactly by anyone
   running the code.

---

## Part 7 - The most important finding in this project

**A high accuracy score does not prove a detector works.**

We tested the trained model against ordinary URLs written by hand, which it
has never seen. The result:

| Test group | Result |
|---|---|
| Normal sites, usual form (`https://www.site.tld`) | 1 of 10 wrongly flagged |
| Normal sites without `www.` (`https://site.tld`) | **3 of 3 wrongly flagged** |
| Normal sites with a page (`/wiki/Phishing`) | **3 of 3 wrongly flagged** |
| Obviously fake phishing links | **6 of 6 correctly caught** |

It still catches every fake link. But it flags ordinary websites like
`https://google.com/` and `https://en.wikipedia.org/wiki/Phishing` as
phishing. At 96% F1.

### Why?

We measured the dataset directly:

| Question about the 134,850 legitimate URLs | Answer |
|---|---|
| How many have no subdomain (no `www.`)? | **0** |
| How many have a non-empty path? | **0** |

**Every single legitimate URL in the dataset is a bare
`https://www.something.tld` homepage.** Not one exception out of 134,850.

So a model can score 96% by learning just two shortcuts:

1. **No `www.` → phishing**
2. **Any path after the domain → phishing**

Both fit the dataset perfectly. Neither describes how phishing works. This is
a property of *how the dataset was collected*, not a coding mistake.

### Why the test set did not catch this

This is the key lesson. The test rows come from the same collection process as
the training rows, so a test score measures *"how well does the model fit this
dataset"* - not *"will this work on the internet"*. **A test set drawn from the
same biased distribution cannot reveal a bias in that distribution.** You have
to probe the model with data from outside it.

### Why we did not just "fix" it

The only way to make the model handle bare domains and deep links would be to
add legitimate examples that we made up ourselves. That is fabricating data,
and it would make every number in this report meaningless. Instead we:

- found and measured the problem,
- fixed the one part that *was* a real bug (the canonicalisation bug in Part 3),
- disclosed the remaining limitation in the app, the README, the report and
  the evaluation output,
- and listed "use a better dataset" as the first item of Future Scope.

That is a stronger research contribution than a slightly higher accuracy
number, and it is much better to discover this yourself than to have an
examiner discover it.

Run `python robustness_check.py` to reproduce all of this.

---

## Part 8 - Feature importance, and what it reveals

The Random Forest's own importance scores (summing to 1.0):

| Feature | Importance |
|---|---|
| `has_https` | 0.4353 |
| `num_digits` | 0.1911 |
| `url_length` | 0.1160 |
| `path_length` | 0.0638 |
| `num_hyphens` | 0.0458 |
| `domain_length` | 0.0381 |
| `num_subdirectories` | 0.0344 |
| `num_subdomains` | 0.0277 |
| `num_dots` | 0.0225 |
| `risky_tld` | 0.0171 |
| `has_suspicious_word` | 0.0049 |
| `num_special_chars` | 0.0025 |
| `num_query_params` | 0.0008 |
| `num_underscores` | 0.0001 |
| `has_at_symbol` | 0.0000 |
| `has_punycode` | 0.0000 |
| `is_shortened_url` | 0.0000 |
| `has_ip_address` | 0.0000 |

**Observation 1 - the top two features carry 63% of the decision.**
`has_https` and `num_digits` together. The model discovered on its own that
secure protocol plus a low digit count is a strong signal.

**Observation 2 - why did `has_suspicious_word` score almost zero (0.0049)?**
Because it is **redundant**. By the time the model already knows the URL is
long and digit-heavy, the extra fact that the word "login" appears adds almost
nothing. This is a genuinely useful observation and the kind of thing that
turns a report into real analysis. It suggests a clear improvement: drop the
redundant features, or use a smaller feature set, and check whether accuracy
holds up.

**Observation 3 - four features score exactly 0.0000.**
`has_at_symbol`, `has_punycode`, `is_shortened_url` and `has_ip_address` all
contribute nothing on this dataset. These detect *real* phishing techniques,
so the honest conclusion is that **the dataset does not contain them**. The
URLs were normalised when the dataset was built, stripping out `xn--`
homograph domains and `@`-obfuscated hosts. A feature scoring zero here is
not a useless feature - it is a coverage gap in the data, and it directly
motivates the Future Scope item about adding page-content and domain
reputation features.

---

## Part 9 - Honest limitations

1. **The model reads only text.** It never opens the website. It has no idea
   what is actually on the page.
2. **"Legitimate" does not mean "safe".** It means the link did not match the
   patterns of phishing. We measured 1,196 missed phishing links out of 20,104 -
   about 1 in 17 - and that is on data similar to the training data.
3. **Real-world accuracy is expected to be lower than 0.9673**, because the
   test set shares the training set's biases. Part 7 shows concrete
   failures on ordinary URLs.
4. **Zero-day attacks will get through.** An attacker who knows these 18
   features can build a URL that deliberately scores as normal.
5. **The dataset is a snapshot in time.** Attack patterns change, so the
   model slowly becomes stale and needs retraining.
6. **The hand-written lists are fixed.** They will always be incomplete, and
   they encode our assumptions rather than measured frequencies.
7. **A single decision tree overfits.** This is why the Random Forest exists.
8. **We deliberately left out page content, DNS and WHOIS.** That keeps the
   project simple, but it also leaves real signals unused.

---

## Part 10 - Future scope

- **Use a dataset whose legitimate class includes deep links and bare
  domains.** This is the single highest-value change, and it is the direct
  remedy for the Part 7 finding.
- **Reduce the feature set** to the top few features, based on the redundancy
  observation in Part 8.
- **Add page-content features** (forms, password fields, external links) for
  a stronger model, accepting the extra complexity.
- **Add domain reputation** (age from WHOIS, DNS records, blacklists).
- **Try gradient boosting** (XGBoost, LightGBM) against the Random Forest.
- **Ship it as a browser extension** so every link is checked before a click.
- **Schedule automatic retraining** so the model never goes stale.
- **Expose it as a REST API** for mail servers and chat clients.
- **Extend from URLs to full emails and SMS messages**, where the real danger
  usually lives.

---

## Part 11 - Likely viva questions

**Q. Why did you choose these three models?**
One simple linear baseline, one explainable tree, and one ensemble that fixes
the tree's overfitting. Together they show a progression from simple to
powerful, which is exactly what a comparison study should do.

**Q. Why F1 and not accuracy?**
Because accuracy is misleading when classes are imbalanced. A model that
always answered "Legitimate" would score 57.29% and still miss every phishing
link. F1 punishes both false alarms and missed attacks at the same time.

**Q. Why is phishing class 1 and not 0?**
Because every precision/recall/F1 formula treats class 1 as the class of
interest. Making phishing class 1 means recall literally answers the question
we care about: how many real phishing links did we catch?

**Q. How do you know your results are not just memorisation?**
Because the model was measured on 47,074 URLs it had never seen. On top of
that, `evaluate_model.py` rebuilds the same split and re-measures, and gets
identical numbers to six decimal places.

**Q. Your accuracy is 96.7%. Is that too good to be true?**
No, and this is the question I am most prepared for. I tested the model
against ordinary websites I wrote by hand, and it **got several wrong** - it
flags `https://google.com/` and any deep link as phishing. The reason is that
every one of the 134,850 legitimate URLs in this dataset is a bare
`https://www.…` homepage, so the model learned "no www. means phishing" as a
shortcut. The test set has the same bias, so the score cannot reveal it. This
is documented in `robustness_check.py` and in the report, and the fix is a
better dataset.

**Q. Didn't you just lower your accuracy on purpose? Why?**
Yes, deliberately, and here is the honest reason. Before URL canonicalisation
the F1 was 0.9951. But the model was flagging the real Wikipedia as phishing
with 100% confidence because of a trailing slash, and it was exploiting the
artefact that any URL with a path is phishing. Fixing that dropped the score
to 0.9608. A lower number that reflects a better system is worth more than a
higher number that hides a bug.

**Q. If it says "Legitimate", is the site safe?**
No. That is the single most important limitation. The model reports a
*pattern-based opinion about a string of text*. It never visits the site, so
it cannot confirm safety. This is why the app always shows the disclaimer.

**Q. What is data leakage and how did you avoid it?**
Letting a model see information it should not have, so results look better
than reality. We avoided it by splitting before fitting, keeping the scaler
inside a Pipeline so it only learns from training rows, deleting 425 duplicate
URLs so the same link could not appear in both halves, and fixing
`random_state=42` for reproducibility. We also did **not** tune any
hyperparameters against the test set, because that is itself leakage.

**Q. Why did some features get zero importance? Does that mean they are useless?**
No, and this is one of the more interesting results. `has_punycode`,
`is_shortened_url`, `has_at_symbol` and `has_ip_address` all scored zero.
They detect real phishing techniques, so the conclusion is that this dataset
does not contain those cases - the URLs were normalised during collection.
A feature scoring zero on one dataset is a coverage gap in the data, not a
useless feature. It is the first thing I would fix with better data.

**Q. What would happen with a brand-new phishing site?**
Probably caught, because the features describe how attackers build links in
general - we caught 6 out of 6 obvious fakes. But not guaranteed. A skilled
attacker who knows our 18 features could deliberately craft a URL that looks
clean.

**Q. Why scikit-learn 1.6.0 and not the newest version?**
Newer scikit-learn releases ship compiled files that are blocked by the
Application Control policy on some Windows machines, producing
`ImportError: DLL load failed while importing _criterion`. Pinning 1.6.0 keeps
the project reproducible and working everywhere. It is pinned in
`requirements.txt` with a comment explaining why.
