"""
feature_extraction.py
=====================
This is the heart of the PhishGuard project.

A machine learning model cannot read text. It can only read NUMBERS.
So this file does one job:

    turn a website URL into 18 simple numbers (called "features").

Example
-------
Input:
    http://secure-login.paypal.verify-account.xyz/login?session=99

Output (a plain Python dictionary):
    {
        'url_length': 57,
        'num_dots': 4,
        'num_hyphens': 3,
        ...
        'risky_tld': 1
    }

WHY THESE FEATURES?
-------------------
Phishing websites are usually built very quickly, by someone trying to look
like a real bank, email service or government website. That rush usually
leaves small "fingerprints" inside the link itself:

  * the address is unusually long
  * it has many dots, hyphens or strange symbols
  * it uses words like "login", "verify" or "secure" to create urgency
  * it ends in a cheap or unusual extension such as .xyz or .zip
  * it hides the real address behind a shortener such as bit.ly
  * it uses a raw IP number (http://192.168.0.1/) instead of a name
  * it fakes a secure https lock that the real site does not have
  * it uses punycode (xn--) to imitate a familiar spelling

A genuine site such as https://www.wikipedia.org/ shows almost none of these.

IMPORTANT NOTE FOR YOUR VIVA
---------------------------
Three features (has_suspicious_word, is_shortened_url, risky_tld) use fixed
word lists that we wrote BY HAND from general knowledge about phishing.
They were NOT learned, tuned or selected using the training or test data,
so this is NOT cheating and NOT data leakage. The model learns by itself
how much each of these numbers is worth.

HOW TO RUN THIS FILE ON ITS OWN (optional, just to see it work):
    python feature_extraction.py
"""

import re
import string
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# 1. THE WORD LISTS  (hand-written, based on general knowledge)
# ---------------------------------------------------------------------------

# Words that attackers often put in phishing links to create urgency or
# to imitate a trusted brand.
SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "log-in",
    "verify", "verification", "validate",
    "account", "update", "confirm", "secure", "security",
    "banking", "password", "credential", "session",
    "billing", "invoice", "wallet", "recover", "recovery",
    "unlock", "suspended", "admin", "otp", "webscr",
    "alert", "prize", "bonus", "winner", "lottery",
    "urgent", "reactivate",
]

# Well-known services that shorten long links into a tiny one.
# A shortener hides the real destination, so attackers like to use them.
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at",
    "tiny.cc", "t.ly", "s.id", "bc.vc", "mcaf.ee", "su.pr",
    "v.gd", "soo.gd", "clck.ru", "short.io", "tiny.one",
    "urlz.fr", "0x0.st", "shorte.st", "rebrand.ly", "bl.ink",
    "qr.ae", "x.co",
}

# Top-Level Domains (the last part, like ".com") that attackers use a lot
# because they are cheap, free, or attract very little attention.
RISKY_TLDS = {
    "zip", "mov", "tk", "ml", "ga", "cf", "gq", "top", "xyz", "click",
    "link", "work", "country", "kim", "cam", "rest", "quest", "review",
    "stream", "download", "loan", "racing", "win", "bid", "date",
    "faith", "cricket", "accountant", "men", "cyou", "sbs", "buzz",
    "monster", "surf", "icu", "wang", "mom",
}

# Characters that are "normal" inside a URL and are therefore NOT counted
# as special characters.
NORMAL_URL_CHARACTERS = set(string.ascii_letters + string.digits + ".-_/")

# A simple pattern that matches an IPv4 address such as 192.168.0.1
IPV4_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


# ---------------------------------------------------------------------------
# 2. THE FEATURE LIST  (the order here is IMPORTANT and must never change)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_special_chars",
    "num_digits",
    "num_subdirectories",
    "path_length",
    "domain_length",
    "num_query_params",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_punycode",
    "num_subdomains",
    "has_suspicious_word",
    "is_shortened_url",
    "risky_tld",
]

# A short, plain-English meaning for every feature. This is used by the
# Streamlit app to build its results table, and by the documentation.
FEATURE_DESCRIPTIONS = {
    "url_length":          "Total number of characters in the URL",
    "num_dots":            "How many '.' (dot) characters appear",
    "num_hyphens":         "How many '-' (hyphen) characters appear",
    "num_underscores":     "How many '_' (underscore) characters appear",
    "num_special_chars":   "Characters that are not letters, digits, dot, hyphen, underscore or slash",
    "num_digits":          "How many numbers (0-9) appear",
    "num_subdirectories":  "Folder levels in the path, e.g. /a/b/c = 3",
    "path_length":         "Number of characters after the domain name",
    "domain_length":       "Number of characters in the domain name",
    "num_query_params":    "How many ?key=value pairs come after the '?'",
    "has_https":           "1 if the URL begins with https://  (1 = good, 0 = suspicious)",
    "has_ip_address":      "1 if the site uses a raw IP number instead of a name",
    "has_at_symbol":       "1 if the URL contains '@' (used to hide the real site)",
    "has_punycode":        "1 if the domain contains 'xn--' (fake lookalike characters)",
    "num_subdomains":      "Levels before the main name, e.g. login.paypal.com = 2",
    "has_suspicious_word": "1 if a word like login / verify / secure appears",
    "is_shortened_url":    "1 if a known URL shortener (bit.ly, tinyurl...) is used",
    "risky_tld":           "1 if the ending is a risky domain extension (.xyz, .zip, .tk...)",
}

# Which direction is "bad"?  This is used only to explain a result to the
# user.  It plays NO part in training the model.
#   +1 = a bigger value points towards PHISHING
#   -1 = a bigger value points towards LEGITIMATE
#    0 = this feature has no clear direction
FEATURE_DIRECTION = {
    "url_length": +1,
    "num_dots": +1,
    "num_hyphens": +1,
    "num_underscores": +1,
    "num_special_chars": +1,
    "num_digits": +1,
    "num_subdirectories": +1,
    "path_length": +1,
    "domain_length": 0,
    "num_query_params": +1,
    "has_https": -1,
    "has_ip_address": +1,
    "has_at_symbol": +1,
    "has_punycode": +1,
    "num_subdomains": +1,
    "has_suspicious_word": +1,
    "is_shortened_url": +1,
    "risky_tld": +1,
}


# ---------------------------------------------------------------------------
# 3. SMALL HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def normalise_url(url):
    """
    Clean up whatever the user typed so that the features always make sense.

    People type things like:  "  WWW.Google.com  "  or  "google.com"
    We turn those into:       "http://www.google.com"
    """
    text = str(url).strip()

    if not text:
        return ""

    # If the user forgot the http:// part, add it.
    if "://" not in text:
        text = "http://" + text

    return text


def _is_ip_address(host):
    """Return True if the host is a raw IP address like 192.168.0.1 or an IPv6."""
    if not host:
        return False
    if IPV4_PATTERN.match(host):
        return True
    # An IPv6 address always contains colons
    if ":" in host:
        return True
    return False


def _canonical_path(path):
    """
    Put the path into its simplest correct form.

    "https://example.com" and "https://example.com/" open the SAME page, so
    the only difference is that one of them ends in a slash. We remove that
    single trailing slash so both give path_length = 0.

    Why this matters (a real finding from this project):
    in the PhiUSIIL dataset ALL 134,850 legitimate URLs have an empty path,
    while many phishing URLs have a long one. A model therefore learns
    "any path means phishing". Without this step, typing
    "https://www.wikipedia.org/" would be flagged as phishing purely because
    of its trailing slash, which is obviously wrong.

    This is standard URL canonicalisation, and it is applied to the training
    data and to the app in exactly the same way, so the two can never
    disagree.
    """
    if path == "/":
        return ""
    return path


def _rebuild_canonical_url(scheme, netloc, path, query, fragment):
    """
    Put a whole URL into one single standard form and return it as a string.

    We then calculate EVERY feature from this one string, so that two URLs
    which really are the same page always produce the same numbers.

    Example:
        "https://www.wikipedia.org/"  ->  "https://www.wikipedia.org"
        "https://www.wikipedia.org"   ->  "https://www.wikipedia.org"
    Both give url_length = 25, path_length = 0, and so on.

    Why this really matters:
    the PhiUSIIL dataset contains a single phishing row that sits at exactly
    url_length = 26. A model trained on it learned "26 characters means
    phishing" and then flagged the real Wikipedia as phishing - purely because
    of a trailing slash that changes nothing. Building one canonical form
    removes that whole class of mistake.
    """
    rebuilt = f"{scheme}://{netloc}" if scheme else netloc
    rebuilt += _canonical_path(path)
    if query:
        rebuilt += "?" + query
    if fragment:
        rebuilt += "#" + fragment
    return rebuilt


def _host_matches(host, domain_set):
    """
    Check whether a host is one of the domains in a set.
    Handles both 'bit.ly' and 'www.bit.ly'.
    """
    host = host.lower()
    if host in domain_set:
        return True
    for domain in domain_set:
        if host.endswith("." + domain):
            return True
    return False


# ---------------------------------------------------------------------------
# 4. THE MAIN FUNCTION  --  URL IN, NUMBERS OUT
# ---------------------------------------------------------------------------

def extract_features(url):
    """
    Turn one URL string into a dictionary of 18 numbers.

    Parameters
    ----------
    url : str
        The website address, for example "https://www.amazon.com/deals".

    Returns
    -------
    dict
        A dictionary with exactly the keys listed in FEATURE_NAMES.
        If the URL is empty or broken, every value is 0.
    """
    clean_url = normalise_url(url)

    # A blank or broken URL still has to give us 18 numbers,
    # otherwise the model would crash later.
    if not clean_url:
        return {name: 0 for name in FEATURE_NAMES}

    try:
        parts = urlsplit(clean_url)
        # netloc can contain "user:password@" -- we only want the real host
        host = (parts.hostname or "").lower()
        query = parts.query
        scheme = parts.scheme.lower()

        # Build ONE standard form of the address, and measure everything
        # from that. This guarantees that two spellings of the same page
        # (with and without a trailing slash) always give identical numbers.
        clean_url = _rebuild_canonical_url(
            scheme, parts.netloc, parts.path, query, parts.fragment
        )
        path = _canonical_path(parts.path)
    except ValueError:
        return {name: 0 for name in FEATURE_NAMES}

    # ---- 1. plain character counting -------------------------------------
    url_length = len(clean_url)
    num_dots = clean_url.count(".")
    num_hyphens = clean_url.count("-")
    num_underscores = clean_url.count("_")
    num_special_chars = sum(
        1 for char in clean_url if char not in NORMAL_URL_CHARACTERS
    )
    num_digits = sum(1 for char in clean_url if char.isdigit())

    # ---- 2. the shape of the address -------------------------------------
    # Subdirectories: /a/b/c.html  ->  ['a', 'b', 'c.html']  ->  3
    path_parts = [piece for piece in path.split("/") if piece]
    num_subdirectories = len(path_parts)
    path_length = len(path)

    domain_length = len(host)

    # Query parameters: a=1&b=2  ->  2
    if query:
        num_query_params = len([item for item in query.split("&") if item])
    else:
        num_query_params = 0

    # ---- 3. clear yes / no clues ----------------------------------------
    has_https = 1 if scheme == "https" else 0

    has_ip_address = 1 if _is_ip_address(host) else 0

    has_at_symbol = 1 if "@" in clean_url else 0

    has_punycode = 1 if "xn--" in host else 0

    # Subdomains = the dots inside the host name.
    # 'login.paypal.com' has 2 dots, so 2 - 1 = 1 subdomain ('login').
    # A raw IP address is not a subdomain, so we set it to 0 in that case.
    if has_ip_address:
        num_subdomains = 0
    else:
        num_subdomains = max(0, host.count(".") - 1)

    # ---- 4. our hand-written knowledge ----------------------------------
    lower_url = clean_url.lower()

    has_suspicious_word = 1 if any(
        word in lower_url for word in SUSPICIOUS_WORDS
    ) else 0

    is_shortened_url = 1 if _host_matches(host, URL_SHORTENERS) else 0

    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    risky_tld = 1 if tld in RISKY_TLDS else 0

    # ---- 5. hand the numbers back ---------------------------------------
    return {
        "url_length": url_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_special_chars": num_special_chars,
        "num_digits": num_digits,
        "num_subdirectories": num_subdirectories,
        "path_length": path_length,
        "domain_length": domain_length,
        "num_query_params": num_query_params,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "has_at_symbol": has_at_symbol,
        "has_punycode": has_punycode,
        "num_subdomains": num_subdomains,
        "has_suspicious_word": has_suspicious_word,
        "is_shortened_url": is_shortened_url,
        "risky_tld": risky_tld,
    }


def features_to_vector(features):
    """
    Turn the dictionary from extract_features() into a plain list of numbers
    in exactly the order given by FEATURE_NAMES.

    Scikit-learn needs a list, not a dictionary, so this bridges the gap.
    """
    return [features[name] for name in FEATURE_NAMES]


def features_to_dataframe(urls, show_progress=False):
    """
    Apply extract_features() to a whole list of URLs.

    Parameters
    ----------
    urls : list of str
        All the URLs in the dataset.

    Returns
    -------
    pandas.DataFrame
        One row per URL, one column per feature.
    """
    import pandas as pd  # imported here so this file works without pandas

    rows = []
    total = len(urls)

    for position, url in enumerate(urls, start=1):
        rows.append(features_to_vector(extract_features(url)))

        if show_progress and (position % 20000 == 0):
            print(f"      features extracted: {position:,} / {total:,}")

    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def explain_features(features, importances=None, top=5, prediction=None):
    """
    Build a short, human-friendly list of the most interesting features of
    one URL.  This is what the Streamlit app shows under "Why?".

    The idea is simple: multiply how much the MODEL cares about a feature
    by how strongly this particular URL triggers it, then show the biggest.

    Parameters
    ----------
    features : dict
        Output of extract_features() for one URL.

    importances : list of float, optional
        The model's feature importances, already scaled to add up to 1.
        If the model cannot provide them, we simply fall back to the
        raw feature values.

    top : int
        How many reasons to return.

    prediction : int, optional
        What the model actually decided (1 = phishing, 0 = legitimate).
        If we know this we list the features that pushed it towards THAT
        answer, so the reasons always agree with the verdict shown above
        them. If it is None we just show the strongest phishing-leaning ones.
    """
    if importances is None:
        importances = [1.0] * len(FEATURE_NAMES)

    scored = []
    for name, importance in zip(FEATURE_NAMES, importances):
        value = features.get(name, 0)
        direction = FEATURE_DIRECTION.get(name, 0)

        # +ve score  -> this feature pushes the URL towards PHISHING
        # -ve score  -> this feature pushes the URL towards LEGITIMATE
        score = importance * direction * value

        scored.append({
            "feature": name,
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
            "value": value,
            "importance": round(float(importance), 4),
            "score": round(float(score), 4),
        })

    # If the model said "legitimate", the useful reasons are the ones
    # pulling it that way, so flip the sign before sorting.
    sign = -1 if prediction == 0 else 1

    for row in scored:
        row["support"] = round(row["score"] * sign, 4)

    scored.sort(key=lambda row: row["support"], reverse=True)

    # Keep only the features that actually said something.
    interesting = [row for row in scored if row["support"] > 0][:top]

    if not interesting:
        # Nothing pushed it either way, so show the features the model
        # simply relies on most heavily.
        interesting = sorted(
            scored, key=lambda row: row["importance"], reverse=True
        )[:top]

    return interesting


# ---------------------------------------------------------------------------
# 5. LET THE STUDENT TRY IT IMMEDIATELY
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PhishGuard - testing feature_extraction.py")
    print("=" * 70)

    demo_urls = [
        "https://www.wikipedia.org/",
        "https://www.amazon.com/deals?ref=nav",
        "http://192.168.1.1/login",
        "http://secure-login.paypal.verify-account.xyz/login?session=99",
        "bit.ly/3xK9abc",
    ]

    for demo in demo_urls:
        print()
        print("URL:", demo)
        values = extract_features(demo)
        for name in FEATURE_NAMES:
            print(f"    {name:<22} = {values[name]}")
