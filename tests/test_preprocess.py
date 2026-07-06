import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_text, handcrafted_features


def test_clean_text_strips_urls():
    result = clean_text("Xem thêm tại https://example.com/bai-viet ngay!")
    assert "https://" not in result, f"URL not stripped: {result}"


def test_clean_text_caps_repeated_punctuation():
    result = clean_text("Tin sốc!!!!!!")
    assert "!!!!" not in result, f"Punctuation not capped: {result}"
    assert "!!!" in result, f"Expected !!! preserved: {result}"


def test_clean_text_preserves_case():
    result = clean_text("BÁO ĐỘNG KHẨN CẤP")
    assert result == "BÁO ĐỘNG KHẨN CẤP", f"Case should be preserved: {result}"


def test_caps_ratio_reflects_case_after_cleaning():
    # Regression test: clean_text() used to lowercase before
    # handcrafted_features() ever saw the text, so caps_ratio was 0
    # for every article regardless of input.
    cleaned = clean_text("CẢNH BÁO khẩn cấp")
    feats = handcrafted_features(cleaned)
    assert feats["caps_ratio"] > 0, f"Expected nonzero caps_ratio, got {feats['caps_ratio']}"


def test_handcrafted_features_counts_exclamations_and_citations():
    text = "Sốc!! Theo nguồn tin, đây là sự thật!!"
    feats = handcrafted_features(text)
    assert feats["exclamation_count"] == 4, f"Expected 4, got {feats['exclamation_count']}"
    assert feats["citation_count"] == 1, f"Expected 1 (theo), got {feats['citation_count']}"


def test_handcrafted_features_handles_empty_string():
    feats = handcrafted_features("")
    assert feats["word_count"] == 1, f"Empty string should default word_count to 1, got {feats['word_count']}"
    assert feats["char_count"] == 0