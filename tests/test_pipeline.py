"""Smoke tests for the Vietnamese Fake News Detector pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import clean_text, handcrafted_features


def test_clean_text_strips_urls():
    result = clean_text("Đọc ngay: https://example.com tin tức mới nhất!")
    assert "https://" not in result
    assert "tin tức mới nhất" in result


def test_clean_text_caps_repeated_punctuation():
    result = clean_text("SỐC!!!!! Phát hiện bí mật?????")
    assert result.count("!") == 3
    assert result.count("?") == 3


def test_handcrafted_features_returns_all_keys():
    feats = handcrafted_features("Bác sĩ tiết lộ bí mật chữa khỏi ung thư 100%!")
    expected = {
        "char_count", "word_count", "exclamation_count", "question_count",
        "caps_ratio", "sensational_count", "citation_count",
        "avg_word_len", "digit_ratio",
    }
    assert set(feats.keys()) == expected


def test_handcrafted_features_exclamation_count():
    feats = handcrafted_features("SỐC! Phát hiện thuốc chữa ung thư 100% từ lá cây rừng!")
    assert feats["exclamation_count"] == 2


def test_handcrafted_features_sensational_keywords():
    # "bí mật", "chữa khỏi", "100%" are all in SENSATIONAL_KEYWORDS
    feats = handcrafted_features("bí mật chữa khỏi ung thư 100%")
    assert feats["sensational_count"] >= 2