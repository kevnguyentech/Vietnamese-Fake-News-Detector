"""
TF-IDF + Logistic Regression baseline classifier.

WHY BUILD A BASELINE BEFORE PHOBERT:
With only ~260 examples, a 135M-parameter transformer has enormous
capacity relative to the data. It *can* overfit in a way that looks
great on one lucky 80/20 split and terrible in production. TF-IDF +
LogReg is fast, interpretable, and surprisingly competitive on small
corpora. If PhoBERT (train_phobert.py) can't beat this by a clear
margin, that's a real finding — the dataset is too small for the
transformer's capacity to pay off yet, and adding more data would
matter more than switching models.

WHY STRATIFIED K-FOLD AND NOT A SINGLE SPLIT:
With 260 examples, a single 80/20 split gives ~52 test articles.
One unlucky draw shifts reported F1 by ±10+ points for reasons
unrelated to the model. 5-fold stratified CV: five non-overlapping
test folds each holding ~52 articles, each fold keeps the Fake/Real
ratio balanced. We report mean ± std across folds — that ± std is
the number that tells you whether you can trust the mean.

WHAT I COMBINE:
  - TF-IDF on word-segmented text (unigrams + bigrams, 10k features)
  - Handcrafted features (exclamation count, caps ratio, sensational
    keyword count, citation marker count, text length, digit ratio…)
  These get stacked horizontally and passed to LogReg together.
  The handcrafted features pick up patterns TF-IDF misses (a 100-word
  article vs a 1000-word one has the same vocabulary but very different
  length), and TF-IDF picks up patterns the handcrafted features miss.

Run:
    python src/baseline.py
"""

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import (
    SEGMENTED_CSV, BASELINE_MODEL_FILE, LABEL_NAMES, RANDOM_SEED,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    N_FOLDS, MODELS_DIR, LOGREG_MAX_ITER, LOGREG_C, LOGREG_CLASS_WEIGHT,
)

HANDCRAFTED_COLS = [
    "char_count", "word_count", "exclamation_count", "question_count",
    "caps_ratio", "sensational_count", "citation_count",
    "avg_word_len", "digit_ratio",
]


def make_pipeline() -> dict:
    """
    Returns a dict of components instead of an sklearn Pipeline so we
    can inspect each piece separately — see get_top_features() below.
    A proper sklearn Pipeline wrapping all of this is also possible but
    makes coefficient extraction more awkward.
    """
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        analyzer="word",
        token_pattern=r"\S+",       # keep Vietnamese underscore tokens
    )
    scaler = StandardScaler()
    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        class_weight=LOGREG_CLASS_WEIGHT,
        C=LOGREG_C,
        random_state=RANDOM_SEED,
    )
    return {"tfidf": tfidf, "scaler": scaler, "clf": clf}


def build_X(df, tfidf, scaler, fit=False):
    """
    Stack TF-IDF matrix + scaled handcrafted features horizontally.
    fit=True during training (fit_transform), False during eval (transform only).
    """
    if fit:
        X_tfidf = tfidf.fit_transform(df["text_seg"])
        X_hand  = scaler.fit_transform(df[HANDCRAFTED_COLS].fillna(0))
    else:
        X_tfidf = tfidf.transform(df["text_seg"])
        X_hand  = scaler.transform(df[HANDCRAFTED_COLS].fillna(0))

    return hstack([X_tfidf, csr_matrix(X_hand)])


def cv_eval(df: pd.DataFrame) -> list:
    """Run 5-fold stratified CV, return per-fold metrics."""
    y = df["label"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), 1):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        comps = make_pipeline()
        X_train = build_X(train_df, comps["tfidf"], comps["scaler"], fit=True)
        X_val   = build_X(val_df,   comps["tfidf"], comps["scaler"], fit=False)

        comps["clf"].fit(X_train, y[train_idx])
        preds = comps["clf"].predict(X_val)

        report = classification_report(
            y[val_idx], preds,
            labels=[0, 1], target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        )
        macro_f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
        results.append({
            "fold":           fold,
            "n_train":        len(train_idx),
            "n_val":          len(val_idx),
            "accuracy":       report["accuracy"],
            "macro_f1":       macro_f1,
            "real_f1":        report["Real"]["f1-score"],
            "fake_f1":        report["Fake"]["f1-score"],
        })
        print(f"fold {fold}: n_train={len(train_idx):3d}  n_val={len(val_idx):2d}  "
              f"accuracy={report['accuracy']:.3f}  macro_f1={macro_f1:.3f}  "
              f"Real_f1={report['Real']['f1-score']:.3f}  Fake_f1={report['Fake']['f1-score']:.3f}")

    return results


def get_top_features(tfidf: TfidfVectorizer, clf: LogisticRegression, n: int = 15):
    """
    LogReg coefficients directly tell you which words push toward Fake
    (positive coefficient) or Real (negative coefficient). This is the
    interpretability win that makes a baseline worth showing even when
    you also have PhoBERT.
    """
    # Feature names: TF-IDF vocab + handcrafted column names
    tfidf_names = tfidf.get_feature_names_out().tolist()
    all_names   = tfidf_names + HANDCRAFTED_COLS
    coefs       = clf.coef_[0]

    top_fake  = [(all_names[i], coefs[i]) for i in np.argsort(coefs)[-n:][::-1]]
    top_real  = [(all_names[i], coefs[i]) for i in np.argsort(coefs)[:n]]
    return top_fake, top_real


def main():
    df = pd.read_csv(SEGMENTED_CSV)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    print(f"Loaded {len(df)} articles  (Fake={sum(df.label==1)}  Real={sum(df.label==0)})")
    print(f"\n--- {N_FOLDS}-fold Stratified CV ---")

    results = cv_eval(df)

    # Summary
    macro_f1s = [r["macro_f1"] for r in results]
    accs      = [r["accuracy"]  for r in results]
    print(f"\nCV summary:")
    print(f"  accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  macro_f1:  {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")

    # Train final model on all data — this is what predict.py loads.
    # We already know its generalisation from CV; now we give it the
    # full dataset so its vocabulary is as large as possible.
    print(f"\n--- Final model (trained on all {len(df)} examples) ---")
    comps = make_pipeline()
    X_all = build_X(df, comps["tfidf"], comps["scaler"], fit=True)
    comps["clf"].fit(X_all, df["label"].values)

    top_fake, top_real = get_top_features(comps["tfidf"], comps["clf"])

    print("\nTop words -> FAKE:")
    for word, coef in top_fake:
        print(f"  {word:30s}  {coef:+.3f}")
    print("\nTop words -> REAL:")
    for word, coef in top_real:
        print(f"  {word:30s}  {coef:+.3f}")

    # Save everything predict.py needs
    joblib.dump({
        "tfidf":             comps["tfidf"],
        "scaler":            comps["scaler"],
        "clf":               comps["clf"],
        "handcrafted_cols":  HANDCRAFTED_COLS,
        "label_names":       LABEL_NAMES,
        "cv_results":        results,
    }, BASELINE_MODEL_FILE)
    print(f"\nSaved model -> {BASELINE_MODEL_FILE}")

if __name__ == "__main__":
    main()
