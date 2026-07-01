"""
The actual tool - run this on any Vietnamese text to get a fake-news
prediction with confidence and an explanation of which words drove it.

Usage:
    # Paste text directly
    python src/predict.py "SỐC! Phát hiện thuốc chữa ung thư 100% từ lá cây rừng"

    # Pipe from a file
    cat article.txt | python src/predict.py

    # Use PhoBERT instead of the TF-IDF baseline (requires train_phobert.py first)
    python src/predict.py --model phobert "your text here"

The TF-IDF model is the default because it (a) runs instantly, (b) works
without a GPU, and (c) is almost as accurate as PhoBERT on this dataset size.
More importantly, it can explain its reasoning - see the "Why?" section of
the output, which lists the words that most strongly pushed the prediction.
PhoBERT can't do that without additional SHAP/LIME post-processing.
"""

import argparse
import sys
import re
import warnings

import numpy as np
import pandas as pd

from config import (
    BASELINE_MODEL_FILE, PHOBERT_DIR, LABEL_NAMES, TOKENIZER,
    SENSATIONAL_KEYWORDS, CITATION_MARKERS,
)
from preprocess import clean_text, get_segmenter, handcrafted_features


def predict_tfidf(text: str) -> dict:
    import joblib
    from scipy.sparse import hstack, csr_matrix

    bundle = joblib.load(BASELINE_MODEL_FILE)
    tfidf  = bundle["tfidf"]
    scaler = bundle["scaler"]
    clf    = bundle["clf"]
    hcols  = bundle["handcrafted_cols"]

    cleaned  = clean_text(text)
    segment  = get_segmenter(TOKENIZER)
    segmented = segment(cleaned)

    hf = handcrafted_features(cleaned)
    X_tfidf = tfidf.transform([segmented])
    X_hand  = scaler.transform(pd.DataFrame([hf])[hcols].fillna(0))
    X = hstack([X_tfidf, csr_matrix(X_hand)])

    proba = clf.predict_proba(X)[0]
    pred  = int(np.argmax(proba))

    # Explanation: top TF-IDF + handcrafted features with non-zero weight
    feature_names = list(tfidf.get_feature_names_out()) + hcols
    coefs  = clf.coef_[0]
    X_dense = X.toarray()[0]
    impact = coefs * X_dense            # element-wise: positive = toward Fake
    top_toward_pred = np.argsort(np.abs(impact))[-10:][::-1]

    reasons = []
    for i in top_toward_pred:
        if abs(impact[i]) > 1e-4:
            direction = "→ Fake" if impact[i] > 0 else "→ Real"
            reasons.append((feature_names[i], impact[i], direction))

    return {
        "prediction": LABEL_NAMES[pred],
        "confidence": proba[pred],
        "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "reasons": reasons,
        "model_used": "TF-IDF + LogReg",
    }


def predict_phobert(text: str) -> dict:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        sys.exit("transformers + torch required for PhoBERT: pip install transformers torch")

    model_path = PHOBERT_DIR / "final"
    if not model_path.exists():
        sys.exit(
            f"PhoBERT model not found at {model_path}.\n"
            "Run: python src/train_phobert.py"
        )

    segment  = get_segmenter(TOKENIZER)
    cleaned  = clean_text(text)
    segmented = segment(cleaned)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    inputs = tokenizer(segmented, return_tensors="pt", truncation=True,
                       padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred  = int(np.argmax(proba))

    return {
        "prediction": LABEL_NAMES[pred],
        "confidence": proba[pred],
        "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "reasons": [],   # PhoBERT explainability needs LIME/SHAP post-processing
        "model_used": "PhoBERT (vinai/phobert-base-v2)",
    }


def print_result(result: dict, text: str):
    pred  = result["prediction"]
    conf  = result["confidence"]
    probs = result["probabilities"]

    bar_width = 30
    fake_bar = "#" * int(probs["Fake"]  * bar_width)
    real_bar = "#" * int(probs["Real"] * bar_width)

    print(f"\n{'─'*55}")
    print(f"  VERDICT:  {pred.upper()}  ({conf*100:.1f}% confidence)")
    print(f"{'─'*55}")
    print(f"  Real  {probs['Real']*100:5.1f}%  {real_bar}")
    print(f"  Fake  {probs['Fake']*100:5.1f}%  {fake_bar}")
    print(f"  Model: {result['model_used']}")

    if result["reasons"]:
        print(f"\n  Why? Top contributing features:")
        for word, impact, direction in result["reasons"][:8]:
            bar = "▓" * min(int(abs(impact) * 5), 20)
            print(f"    {word:25s}  {direction}  {bar}")

    if pred == "Fake":
        print(f"\n  ⚠  This article shows characteristics of misinformation.")
        print(f"     Verify with VnExpress, Tuổi Trẻ, or Nhân Dân before sharing.")
    else:
        print(f"\n  ✓  This article shows characteristics of legitimate news.")
        print(f"     Always verify important claims regardless of prediction.")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="Vietnamese text to classify")
    parser.add_argument("--model", choices=["tfidf", "phobert"], default="tfidf",
                         help="Which model to use (default: tfidf)")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    text = text.strip()
    if len(text) < 5:
        sys.exit("Text too short to classify.")

    print(f"\nInput ({len(text)} chars): {text[:100]}{'…' if len(text)>100 else ''}")

    if args.model == "phobert":
        result = predict_phobert(text)
    else:
        result = predict_tfidf(text)

    print_result(result, text)


if __name__ == "__main__":
    main()
