import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import looks_vietnamese


def test_looks_vietnamese_true_for_diacritic_text():
    assert looks_vietnamese("Cảnh báo khẩn cấp") is True


def test_looks_vietnamese_false_for_plain_english():
    assert looks_vietnamese("Breaking news update today") is False


def test_looks_vietnamese_false_for_empty_string():
    assert looks_vietnamese("") is False


def test_looks_vietnamese_true_for_uppercase_diacritics():
    assert looks_vietnamese("CẢNH BÁO") is True
    