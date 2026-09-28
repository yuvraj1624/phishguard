"""
app.py
======
The PhishGuard web application, built with Streamlit.

WHAT THE USER SEES
------------------
  * A title and a short explanation
  * One input box:   "Enter website URL"
  * One button:      "Check URL"
  * The result:      PHISHING or LEGITIMATE, with a confidence bar,
                     the extracted features, and the reasons behind the
                     decision.

HOW IT WORKS UNDER THE HOOD
---------------------------
  1. The user types a URL.
  2. feature_extraction.py turns it into 18 numbers.
  3. The model saved by train_model.py reads those 18 numbers.
  4. Streamlit displays whatever the model says.

THE MODEL IS NEVER REBUILT HERE. app.py only LOADS the file that
train_model.py saved. That is why training and prediction can never
disagree about which features to use.

HOW TO RUN:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extraction import (FEATURE_DESCRIPTIONS, explain_features,
                                extract_features, features_to_vector)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard - Phishing Website Detector",
    page_icon="shield",
    layout="centered",
)

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(PROJECT_FOLDER, "models", "phishguard_model.joblib")

EXAMPLE_URLS = [
    "https://www.wikipedia.org/",
    "https://www.amazon.com/",
    "http://192.168.0.1/login",
    "http://secure-login.paypal-verify-account.xyz/login?session=9981",
]

DISCLAIMER = (
    "This is a machine learning prediction, not confirmed website safety. "
    "The model reads only the text of the link. It never opens or visits the "
    "website, so it cannot prove that a site is safe. A 'Legitimate' result "
    "means the link did not look unusual - it is not a guarantee. When in "
    "doubt, do not enter passwords or personal details."
)

KNOWN_LIMITATION = (
    "Known limitation: every legitimate URL in the training dataset is a "
    "bare 'https://www.something.tld' homepage. Because of that, this model "
    "tends to flag two things it should not - addresses typed without "
    "'www.' (for example 'github.com'), and ordinary deep links that contain "
    "a page (for example 'en.wikipedia.org/wiki/Phishing'). This is a "
    "property of the dataset, not of the code. Run "
    "'python robustness_check.py' to see the evidence."
)


# ---------------------------------------------------------------------------
# Load the model (only once, then reuse it)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    """Read the trained model from disk. Cached so it loads only one time."""
    return joblib.load(path)


def main():
    # ---------------- header ----------------
    st.title("PhishGuard - Phishing Website Detector")
    st.markdown(
        """
        This application uses **machine learning** to guess whether a web
        address is **phishing** (a fake website built to steal your details)
        or **legitimate** (an ordinary website).

        It works in three steps:

        1. It reads the link you type.
        2. It converts the link into **18 simple numbers** such as length,
           number of dots, number of sub-folders, and whether it uses `https`.
        3. A trained model looks at those numbers and gives a verdict.
        """
    )

    # ---------------- load model ----------------
    try:
        bundle = load_model(MODEL_FILE)
    except FileNotFoundError:
        st.error("### No trained model found")
        st.write(
            f"PhishGuard looked for the model file here:\n\n"
            f"`{MODEL_FILE}`\n\n"
            f"It does not exist yet. Train it first:\n\n"
            f"```\npython download_dataset.py\npython train_model.py\n```"
        )
        st.stop()
    except Exception as error:                                  # noqa: BLE001
        st.error(f"### The model file could not be loaded\n\n`{error}`")
        st.write("Retrain it with: `python train_model.py`")
        st.stop()

    class_names = bundle.get("class_names", ["Legitimate", "Phishing"])
    metrics = bundle["metrics"]
    model_name = bundle["best_model_name"]

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("Model information")
        st.write(f"**Selected model:** {model_name}")
        st.write(f"**Selection rule:** {bundle['selection_metric']}")
        st.write(f"**Features used:** {len(bundle['feature_names'])}")
        st.write(f"**Training rows:** {bundle['training_rows']:,}")
        st.write(f"**Dataset:** {bundle['dataset_name']}")
        st.write(f"**Trained:** {bundle['trained_at']}")

        st.subheader("Measured performance")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
                "Score": [f"{metrics['accuracy']:.4f}",
                          f"{metrics['precision']:.4f}",
                          f"{metrics['recall']:.4f}",
                          f"{metrics['f1']:.4f}"],
            }),
            hide_index=True,
        )
        st.caption("Measured on a 20% test set the model never trained on.")

    # ---------------- input ----------------
    st.subheader("Check a website")

    entered_url = st.text_input(
        "Enter website URL",
        value="",
        placeholder="https://www.example.com",
    )

    left, right, _ = st.columns([1, 1, 4])
    check_clicked = left.button("Check URL", type="primary")
    if right.button("Clear"):
        st.rerun()

    # Example buttons so the app is easy to demonstrate.
    st.caption("No URL to hand? Try one of these demo strings:")
    example_row = st.columns(4)
    chosen_example = None
    for column, example in zip(example_row, EXAMPLE_URLS):
        if column.button(example.replace("https://", "")
                                  .replace("http://", "")[:26],
                         key=example):
            chosen_example = example

    if chosen_example and entered_url.strip() == "":
        entered_url = chosen_example

    # ---------------- prediction ----------------
    if check_clicked:
        url_to_check = entered_url.strip()

        if not url_to_check:
            st.warning("Please type a website address first.")
            st.stop()

        if "." not in url_to_check:
            st.warning(
                "That does not look like a full web address - it has no dot. "
                "For example: https://www.example.com"
            )

        # --- 1. turn the URL into numbers ---
        features = extract_features(url_to_check)
        vector = np.array(
            features_to_vector(features)
        ).reshape(1, -1)

        # --- 2. ask the model ---
        model = bundle["model"]
        predicted_class = int(model.predict(vector)[0])
        verdict = class_names[predicted_class]

        phishing_probability = None
        if hasattr(model, "predict_proba"):
            phishing_probability = float(model.predict_proba(vector)[0][1])

        st.divider()

        # --- 3. show the verdict ---
        if predicted_class == 1:
            st.error(f"## PHISHING - this link looks dangerous")
        else:
            st.success(f"## LEGITIMATE - this link did not look suspicious")

        if phishing_probability is not None:
            left_column, right_column = st.columns(2)
            with left_column:
                st.metric("Model confidence (phishing)",
                          f"{phishing_probability * 100:.1f}%")
                st.progress(phishing_probability)
            with right_column:
                st.metric("Model confidence (legitimate)",
                          f"{(1 - phishing_probability) * 100:.1f}%")
                st.progress(1 - phishing_probability)
        else:
            st.info("This model does not provide a confidence score.")

        st.caption(f"Verdict given by: **{model_name}**")

        # --- 4. why? ---
        reasons = explain_features(features, bundle.get("importances"), top=5,
                                   prediction=predicted_class)

        if reasons:
            st.subheader(f"Why the model leaned towards {verdict.upper()}")
            st.write(
                "**Importance** is how much this model relies on that "
                "feature in general. **Your URL** is what your address "
                "actually scored. The list shows the features that pushed "
                "the model towards the verdict above."
            )
            reason_rows = []
            for reason in reasons:
                reason_rows.append({
                    "Feature": reason["feature"],
                    "What it means": reason["description"],
                    "Your URL": reason["value"],
                    "Model importance": reason["importance"],
                })
            st.dataframe(pd.DataFrame(reason_rows), hide_index=True,
                         use_container_width=True)

        # --- 5. full feature table ---
        with st.expander("Show all 18 extracted features"):
            rows = []
            for name in bundle["feature_names"]:
                rows.append({
                    "Feature": name,
                    "Meaning": FEATURE_DESCRIPTIONS.get(name, ""),
                    "Value": features[name],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

        # --- 6. the warning the user must always see ---
        st.warning(DISCLAIMER)
        with st.expander("Known limitations of this model"):
            st.write(KNOWN_LIMITATION)

    else:
        st.markdown(
            """
            ---
            ### Before you start

            This tool gives an **opinion based on the text of a link**.
            It is a helpful first check, but it is not a security guarantee.
            Always type important addresses yourself instead of clicking
            links from emails or messages.
            """
        )


if __name__ == "__main__":
    main()
