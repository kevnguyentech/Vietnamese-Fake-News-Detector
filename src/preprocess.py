"""
Vietnamese text cleaning + word segmentation.

WHY SEGMENTATION MATTERS:
Vietnamese is a tonal, multi-syllabic language where a single "word"
(meaning unit) is often two or three syllables written with spaces
between them. "học sinh" (student) is one word but looks like two
tokens to a naive whitespace tokenizer - it'll split them apart and
treat "học" and "sinh" as separate features, missing the actual meaning.
Proper segmentation joins multi-syllable words with underscores:
"học_sinh", so TF-IDF sees one meaningful token, not two random ones.

PhoBERT was pretrained on pre-segmented text. Feeding it raw
un-segmented Vietnamese still works (it won't crash) but performs
noticeably worse, because its vocabulary was built from segmented text.

Three tokenizers are supported, ranked by quality:
  1. underthesea — best accuracy, pure Python, no Java needed
  2. pyvi        — slightly lower accuracy, also pure Python
  3. simple      — whitespace split only (fallback, no segmentation)

Run:
    python src/preprocess.py
"""

import re
import sys

import pandas as pd

from config import (
    PROCESSED_CSV, SEGMENTED_CSV, TOKENIZER, SENSATIONAL_KEYWORDS, CITATION_MARKERS
)


# ── cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove noise without erasing linguistic signals.
    Key choices:
    - URLs stripped (carry source-domain info already in source_domain col)
    - HTML tags stripped (some articles have residual markup)
    - Repeated whitespace collapsed
    - Repeated punctuation capped at 3 (!!!! -> !!!, real signal for tone)
    - Case preserved (handcrafted_features() needs real casing for
      caps_ratio; callers that need lowercase, like the segmenter,
      lowercase explicitly at the call site)
    - Emojis kept: in social-media posts they carry real sentiment signal
    """
    if not isinstance(text, str):
        return ""
    # strip URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    # strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # cap repeated punctuation at 3
    text = re.sub(r"([!?.]){4,}", r"\1\1\1", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── segmentation ──────────────────────────────────────────────────────

def get_segmenter(name: str):
    """
    Returns a function: raw_text -> segmented_text.
    Tries the preferred tokenizer, falls back gracefully if unavailable.
    """
    if name == "underthesea":
        try:
            from underthesea import word_tokenize
            def segment_underthesea(text: str) -> str:
                return word_tokenize(text, format="text")
            return segment_underthesea
        except ImportError:
            print("underthesea not installed, falling back to pyvi", file=sys.stderr)
            name = "pyvi"

    if name == "pyvi":
        try:
            import warnings
            from pyvi import ViTokenizer
            def segment_pyvi(text: str) -> str:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return ViTokenizer.tokenize(text)
            return segment_pyvi
        except ImportError:
            print("pyvi not installed, falling back to simple tokenizer", file=sys.stderr)
            name = "simple"

    # simple: no real segmentation, just whitespace normalisation
    def segment_simple(text: str) -> str:
        return text
    return segment_simple


# ── handcrafted features ──────────────────────────────────────────────

def handcrafted_features(text: str) -> dict:
    """
    Explicit features that TF-IDF can't capture on its own.
    These are used both as model features AND as a sanity check:
    if sensational_count has zero importance in SHAP/coefficients,
    the model isn't picking up on what we expected and we should dig in.
    """
    low = text.lower()
    words = low.split()
    n = max(len(words), 1)

    return {
        "char_count":          len(text),
        "word_count":          n,
        "exclamation_count":   text.count("!"),
        "question_count":      text.count("?"),
        "caps_ratio":          sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "sensational_count":   sum(kw in low for kw in SENSATIONAL_KEYWORDS),
        "citation_count":      sum(m in low for m in CITATION_MARKERS),
        "avg_word_len":        sum(len(w) for w in words) / n,
        "digit_ratio":         sum(c.isdigit() for c in text) / max(len(text), 1),
    }


def main():
    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} articles from {PROCESSED_CSV}")

    print(f"Cleaning text …")
    df["text_clean"] = df["text"].apply(clean_text)

    print(f"Segmenting with '{TOKENIZER}' (this takes ~30s for 260 articles) …")
    segment = get_segmenter(TOKENIZER)
    df["text_seg"] = df["text_clean"].str.lower().apply(segment)

    print("Computing handcrafted features …")
    feats = df["text_clean"].apply(handcrafted_features).apply(pd.Series)
    df = pd.concat([df, feats], axis=1)

    df.to_csv(SEGMENTED_CSV, index=False)
    print(f"\nSaved -> {SEGMENTED_CSV}")

    print(f"\nHandcrafted feature means by class:")
    feat_cols = list(feats.columns)
    print(df.groupby("label_name")[feat_cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
