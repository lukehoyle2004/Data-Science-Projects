"""
Email Spam Classifier (Naive Bayes) — cleaned from a Colab notebook export.

What this script does:
- Loads one or more CSV datasets containing email bodies + labels
- Cleans text (lowercase, remove punctuation, remove stopwords, stemming)
- Vectorizes text (Bag-of-Words)
- Trains Multinomial Naive Bayes
- Optionally tunes Laplace smoothing (alpha) and reports metrics

Expected CSV columns (minimum):
- Body  : email text
- Label : 0/1 or ham/spam (script will try to normalize)

Usage examples:
  python email_spam_classifier.py --csv enronSpamSubset.csv --text_col Body --label_col Label
  python email_spam_classifier.py --csv enronSpamSubset.csv lingSpam.csv --tune_alpha

Notes:
- If you haven't downloaded NLTK resources before, the script will attempt to download them.
- This script avoids notebook-only commands (no "!pip install" etc.).
"""

from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB

# --- Optional NLTK (used for stopwords + stemming) ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except Exception as e:  # pragma: no cover
    nltk = None
    stopwords = None
    PorterStemmer = None
    word_tokenize = None


def _ensure_nltk() -> None:
    """Download NLTK resources if needed."""
    if nltk is None:
        raise RuntimeError(
            "NLTK is not installed. Install it with: pip install nltk"
        )
    # These calls are safe if already downloaded
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)


def normalize_labels(y: pd.Series) -> pd.Series:
    """
    Normalize common label formats into 0/1 where 1 = spam.
    Accepts: 0/1, 'spam'/'ham', 'spam'/'not spam', etc.
    """
    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "spam": 1,
        "1": 1,
        "true": 1,
        "yes": 1,
        "ham": 0,
        "not spam": 0,
        "0": 0,
        "false": 0,
        "no": 0,
    }
    mapped = y_str.map(mapping)
    if mapped.isna().any():
        # Fall back: try to cast to int if possible
        try:
            return y_str.astype(int)
        except Exception as e:
            unknown = sorted(set(y_str[mapped.isna()].unique().tolist()))
            raise ValueError(
                f"Unrecognized labels found: {unknown}. "
                "Please normalize labels to 0/1 or ham/spam."
            )
    return mapped.astype(int)


def clean_text_series(text: pd.Series, use_stemming: bool = True) -> pd.Series:
    """Basic NLP preprocessing."""
    _ensure_nltk()

    s = text.fillna("").astype(str).str.lower()
    # remove punctuation
    s = s.apply(lambda t: t.translate(str.maketrans("", "", string.punctuation)))

    sw = set(stopwords.words("english"))  # type: ignore
    s = s.apply(lambda t: " ".join([w for w in t.split() if w not in sw]))

    if use_stemming:
        stemmer = PorterStemmer()  # type: ignore
        s = s.apply(lambda t: " ".join(stemmer.stem(w) for w in word_tokenize(t)))  # type: ignore
    return s


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # Common notebook artifact
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna().drop_duplicates()
    return data


def tune_alpha(
    X, y, alphas: Iterable[float], *, cv_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Grid-search alpha with CV and return sorted results."""
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for a in alphas:
        accs, precs, recs, f1s = [], [], [], []
        for tr_idx, te_idx in kf.split(X):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            nb = MultinomialNB(alpha=a)
            nb.fit(Xtr, ytr)
            pred = nb.predict(Xte)
            m = evaluate(yte.to_numpy(), pred)
            accs.append(m.accuracy)
            precs.append(m.precision)
            recs.append(m.recall)
            f1s.append(m.f1)
        rows.append(
            {
                "alpha": a,
                "accuracy_mean": float(np.mean(accs)),
                "precision_mean": float(np.mean(precs)),
                "recall_mean": float(np.mean(recs)),
                "f1_mean": float(np.mean(f1s)),
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Naive Bayes spam classifier.")
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more CSV files (local paths).",
    )
    parser.add_argument("--text_col", default="Body", help="Text column name.")
    parser.add_argument("--label_col", default="Label", help="Label column name.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tune_alpha", action="store_true", help="Tune Laplace alpha.")
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,0.5,1.0,2.0,5.0",
        help="Comma-separated alpha values to try (used with --tune_alpha).",
    )
    parser.add_argument(
        "--no_stem",
        action="store_true",
        help="Disable stemming (sometimes improves interpretability).",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    data = load_and_merge(csv_paths)

    if args.text_col not in data.columns or args.label_col not in data.columns:
        raise KeyError(
            f"Missing columns. Found {list(data.columns)}; "
            f"expected text_col='{args.text_col}', label_col='{args.label_col}'."
        )

    data[args.label_col] = normalize_labels(data[args.label_col])
    data[args.text_col] = clean_text_series(data[args.text_col], use_stemming=not args.no_stem)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[args.text_col])
    y = data[args.label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    best_alpha = 1.0
    if args.tune_alpha:
        alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
        results = tune_alpha(X_train, y_train, alphas)
        best_alpha = float(results.iloc[0]["alpha"])
        print("Alpha tuning results (top 5 by F1):")
        print(results.head(5).to_string(index=False))
        print(f"\nBest alpha selected: {best_alpha}\n")

    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m = evaluate(y_test.to_numpy(), y_pred)
    print(f"Accuracy : {m.accuracy:.4f}")
    print(f"Precision: {m.precision:.4f}")
    print(f"Recall   : {m.recall:.4f}")
    print(f"F1       : {m.f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
