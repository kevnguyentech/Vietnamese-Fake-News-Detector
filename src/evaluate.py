"""
Loads the saved TF-IDF baseline and produces:
  1. confusion_matrix.png  — where predictions go wrong
  2. top_features.png      — which words/features push toward Fake vs Real
  3. error_analysis.csv    — the 10 most confidently wrong predictions,
                             so you can read the actual text and understand
                             the failure mode (usually ambiguous headlines
                             or short articles with too little signal)

Run:
    python src/evaluate.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import (
    SEGMENTED_CSV, BASELINE_MODEL_FILE, OUTPUTS_DIR, LABEL_NAMES,
)
from baseline import build_X


def load_model():
    return joblib.load(BASELINE_MODEL_FILE)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Predicted vs. Actual\n(out-of-sample CV predictions)")
    fig.tight_layout()
    path = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_features(tfidf, clf, handcrafted_cols, n=20):
    """
    LogReg coefficients for the Fake class (positive = pushes toward Fake).
    Shows top-n features in each direction.
    This is one of the most useful plots for a portfolio: it shows you
    can explain *what* your model learned, not just report an F1 score.
    """
    feature_names = list(tfidf.get_feature_names_out()) + handcrafted_cols
    coefs = clf.coef_[0]

    top_fake_idx = np.argsort(coefs)[-n:][::-1]
    top_real_idx = np.argsort(coefs)[:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Fake side (positive coefs)
    labels_f = [feature_names[i] for i in top_fake_idx]
    vals_f   = [coefs[i]          for i in top_fake_idx]
    ax1.barh(range(n), vals_f[::-1], color="#C0392B")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(labels_f[::-1], fontsize=9)
    ax1.set_xlabel("Logistic Regression coefficient")
    ax1.set_title(f"Top {n} features → FAKE")
    ax1.axvline(0, color="black", linewidth=0.5)

    # Real side (negative coefs)
    labels_r = [feature_names[i] for i in top_real_idx]
    vals_r   = [coefs[i]          for i in top_real_idx]
    ax2.barh(range(n), [abs(v) for v in vals_r[::-1]], color="#27AE60")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels_r[::-1], fontsize=9)
    ax2.set_xlabel("|coefficient| (negative = Real signal)")
    ax2.set_title(f"Top {n} features → REAL")

    fig.suptitle("What the model learned: TF-IDF + handcrafted feature coefficients",
                 fontsize=11)
    fig.tight_layout()
    path = OUTPUTS_DIR / "top_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def error_analysis(df, y_pred, y_proba):
    """
    The most valuable diagnostic: which articles was the model most
    *confidently* wrong about? Reading these reveals failure modes
    that aggregate metrics hide — usually they're short articles with
    almost no text, or articles where the headline is sensational but
    the content is legitimate journalism, or vice versa.
    """
    df2 = df.copy()
    df2["pred"]         = y_pred
    df2["confidence"]   = y_proba.max(axis=1)
    df2["correct"]      = (df2["label"] == df2["pred"])

    wrong = df2[~df2["correct"]].sort_values("confidence", ascending=False)
    cols  = ["id", "label_name", "pred", "confidence", "text"]
    out   = wrong[cols].head(10).copy()
    out["pred_name"] = out["pred"].map({0: "Real", 1: "Fake"})
    out["text_preview"] = out["text"].str[:120].str.replace("\n", " ")
    out = out.drop(columns=["pred", "text"])

    path = OUTPUTS_DIR / "error_analysis.csv"
    out.to_csv(path, index=False)
    print(f"\nTop 10 most confident wrong predictions -> {path}")
    for _, row in out.iterrows():
        print(f"  True={row['label_name']:4s}  Pred={row['pred_name']:4s}  "
              f"Conf={row['confidence']:.2f}  |  {row['text_preview'][:80]}…")


def main():
    bundle = load_model()
    tfidf          = bundle["tfidf"]
    scaler         = bundle["scaler"]
    clf            = bundle["clf"]
    handcrafted_cols = bundle["handcrafted_cols"]
    df = pd.read_csv(SEGMENTED_CSV)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    oos_preds = bundle.get("oos_preds")
    oos_proba = bundle.get("oos_proba")

    oos_valid = (oos_preds is not None and oos_proba is not None
                 and len(oos_preds) == len(df))
    if oos_valid:
        y_pred  = oos_preds
        y_proba = oos_proba
    else:
        if oos_preds is not None and len(oos_preds) != len(df):
            print(f"Warning: bundle has {len(oos_preds)} OOS rows but dataset has "
                  f"{len(df)} rows; re-run baseline.py. Falling back to in-sample.")
        else:
            print("Warning: bundle missing OOS predictions; using in-sample. Re-run baseline.py.")
        X = build_X(df, tfidf, scaler, fit=False, handcrafted_cols=handcrafted_cols)
        y_pred  = clf.predict(X)
        y_proba = clf.predict_proba(X)

    plot_confusion_matrix(df["label"].values, y_pred)
    plot_top_features(tfidf, clf, handcrafted_cols)
    error_analysis(df, y_pred, y_proba)

    cv = bundle.get("cv_results", [])
    if cv:
        macro_f1s = [r["macro_f1"] for r in cv]
        accs      = [r["accuracy"]  for r in cv]
        print(f"\nStored CV performance (from baseline.py):")
        print(f"  accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  macro_f1:  {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")

if __name__ == "__main__":
    main()
