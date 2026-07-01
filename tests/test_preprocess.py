import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_text, handcrafted_features

# Test 1: clean_text strips URLs
result = clean_text("Xem thêm tại https://example.com/bai-viet ngay!")
assert "https://" not in result, f"URL not stripped: {result}"
print("PASS: clean_text strips URLs")

# Test 2: clean_text caps repeated punctuation at 3
result = clean_text("Tin sốc!!!!!!")
assert "!!!!" not in result, f"Punctuation not capped: {result}"
assert "!!!" in result, f"Expected !!! preserved: {result}"
print("PASS: clean_text caps punctuation")

# Test 3: clean_text lowercases
result = clean_text("BÁO ĐỘNG KHẨN CẤP")
assert result == result.lower(), f"Not lowercased: {result}"
print("PASS: clean_text lowercases")

# Test 4: handcrafted_features counts exclamations and citations correctly
text = "Sốc!! Theo nguồn tin, đây là sự thật!!"
feats = handcrafted_features(text)
assert feats["exclamation_count"] == 4, f"Expected 4, got {feats['exclamation_count']}"
assert feats["citation_count"] == 1, f"Expected 1 (theo), got {feats['citation_count']}"
print("PASS: handcrafted_features counts exclamations/citations")

# Test 5: handcrafted_features handles empty string without crashing
feats = handcrafted_features("")
assert feats["word_count"] == 1, f"Empty string should default word_count to 1, got {feats['word_count']}"
assert feats["char_count"] == 0
print("PASS: handcrafted_features handles empty string")

print("\nAll tests passed.")