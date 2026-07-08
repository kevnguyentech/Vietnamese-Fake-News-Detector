import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from baseline import make_pipeline, build_X, cv_eval, HANDCRAFTED_COLS


def test_build_X_shape_matches_tfidf_vocab_plus_handcrafted_cols():
    # Regression check for the tfidf/handcrafted hstack in build_X():
    # if that concatenation ever drops or duplicates a matrix, the
    # column count silently stops matching tfidf vocab + HANDCRAFTED_COLS.
    df = pd.DataFrame({
        "text_seg": [
            "tin tức thật",
            "tin giả mạo",
            "báo chí chính thống",
            "tin đồn thất thiệt",
        ],
        "char_count":         [10, 12, 15, 14],
        "word_count":         [3, 3, 4, 4],
        "exclamation_count":  [0, 1, 0, 2],
        "question_count":     [0, 0, 0, 1],
        "caps_ratio":         [0.0, 0.05, 0.0, 0.1],
        "sensational_count":  [0, 1, 0, 2],
        "citation_count":     [0, 0, 1, 0],
        "avg_word_len":       [3.3, 3.5, 3.6, 3.4],
        "digit_ratio":        [0.0, 0.0, 0.0, 0.0],
    })

    comps = make_pipeline()
    X = build_X(df, comps["tfidf"], comps["scaler"], fit=True)

    expected_cols = len(comps["tfidf"].get_feature_names_out()) + len(HANDCRAFTED_COLS)
    assert X.shape == (len(df), expected_cols), (
        f"build_X shape {X.shape} != (n_rows, tfidf_vocab + "
        f"len(HANDCRAFTED_COLS)) = {(len(df), expected_cols)}"
    )


def test_build_X_explicit_handcrafted_cols_overrides_default():
    # Guard against column drift: if the saved bundle has a different
    # column list than the current module default, build_X must use the
    # bundle's list, not HANDCRAFTED_COLS.
    df = pd.DataFrame({
        "text_seg":          ["tin tức thật", "tin giả mạo"],
        "char_count":        [10, 12],
        "word_count":        [3, 3],
        "exclamation_count": [0, 1],
        "question_count":    [0, 0],
        "caps_ratio":        [0.0, 0.05],
        "sensational_count": [0, 1],
        "citation_count":    [0, 0],
        "avg_word_len":      [3.3, 3.5],
        "digit_ratio":       [0.0, 0.0],
    })
    subset = ["char_count", "word_count"]  # 2 cols, not the full 9

    comps = make_pipeline()
    X = build_X(df, comps["tfidf"], comps["scaler"], fit=True, handcrafted_cols=subset)

    expected_cols = len(comps["tfidf"].get_feature_names_out()) + len(subset)
    assert X.shape[1] == expected_cols


def test_cv_eval_oos_predictions_cover_every_row():
    df = pd.DataFrame({
        "text_seg": [
            "tin tức thật",       "tin giả mạo",
            "báo chí chính thống","tin đồn thất thiệt",
            "tin tức thật sự",    "tin tức giả mạo lan",
            "báo chính xác hôm",  "cảnh báo khẩn cấp",
            "nguồn tin đáng tin",  "tin đồn vô căn cứ",
        ],
        "label":             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "char_count":        [10, 12, 15, 14, 11, 13, 16, 9, 12, 14],
        "word_count":        [3, 3, 4, 4, 3, 4, 4, 3, 3, 4],
        "exclamation_count": [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        "question_count":    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "caps_ratio":        [0.0, 0.05, 0.0, 0.1, 0.0, 0.05, 0.0, 0.1, 0.0, 0.05],
        "sensational_count": [0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
        "citation_count":    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        "avg_word_len":      [3.3, 3.5, 3.6, 3.4, 3.3, 3.5, 3.6, 3.4, 3.3, 3.5],
        "digit_ratio":       [0.0] * 10,
    })

    results, oos_preds, oos_proba = cv_eval(df)

    assert len(oos_preds) == len(df)
    assert oos_proba.shape == (len(df), 2)
    assert np.allclose(oos_proba.sum(axis=1), 1.0, atol=1e-6)