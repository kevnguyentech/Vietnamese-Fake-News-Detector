import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parse_vfnd import extract_text


def test_extract_text_merges_title_and_body_with_sep():
    d = {"title": "SỐC: tin nóng", "text": "Nội dung chi tiết ở đây."}
    assert extract_text(d) == "SỐC: tin nóng [SEP] Nội dung chi tiết ở đây."


def test_extract_text_falls_back_to_title_rss():
    d = {"title_rss": "Tiêu đề RSS", "text": "Nội dung."}
    assert extract_text(d) == "Tiêu đề RSS [SEP] Nội dung."


def test_extract_text_title_only():
    d = {"title": "Chỉ có tiêu đề"}
    assert extract_text(d) == "Chỉ có tiêu đề"


def test_extract_text_body_only():
    d = {"text": "Chỉ có nội dung"}
    assert extract_text(d) == "Chỉ có nội dung"


def test_extract_text_missing_keys_returns_empty_string():
    assert extract_text({}) == ""