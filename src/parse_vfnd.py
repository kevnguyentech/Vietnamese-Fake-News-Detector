"""
Parses the raw VFND folder structure into a single flat CSV.

VFND structure:
  VFND/
    Fake/Article_Contents/VFND_Ac_Fake_*.json
    Fake/Social_Contents/VFND_So_Fake_*.json
    Real/Article_Contents/VFND_Ac_Real_*.json
    Misleading/… (dropped — only 2 files)

Each JSON has 'title' + 'text' fields (articles) or 'text' (social).
We merge title + body into one string the model sees end to end.

Run:
    python src/parse_vfnd.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

from config import VFND_DIR, PROCESSED_CSV, LABEL_MAP


def load_json_safe(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: {path.name}: {e}", file=sys.stderr)
        return {}


def extract_text(d: dict) -> str:
    """
    Merge title + body. '[SEP]' is both a readable delimiter and the
    actual PhoBERT sentence separator token — so this format works
    identically for both TF-IDF and transformer pipelines.
    Title matters a lot: sensational phrasing ("SỐC!", "NÓNG:") is
    the single strongest fake-news signal in Vietnamese tabloid content.
    Body matters too: sourcing patterns ("Bộ Y tế cho biết") vs vague
    attribution ("theo nguồn tin") separate real from fake in the body.
    Dropping either loses signal, so we always use both.
    """
    title = (d.get("title") or d.get("title_rss") or "").strip()
    body  = (d.get("text") or "").strip()
    if title and body:
        return f"{title} [SEP] {body}"
    return title or body


def parse_vfnd(vfnd_dir: Path) -> pd.DataFrame:
    rows = []
    for label_name, label_int in LABEL_MAP.items():
        label_dir = vfnd_dir / label_name
        if not label_dir.exists():
            continue
        for ctype in ["Article_Contents", "Social_Contents"]:
            ctype_dir = label_dir / ctype
            if not ctype_dir.exists():
                continue
            source_type = "article" if "Article" in ctype else "social"
            for path in sorted(ctype_dir.glob("*.json")):
                d = load_json_safe(path)
                if not d:
                    continue
                text = extract_text(d)
                if not text:
                    continue
                rows.append({
                    "id":           path.stem,
                    "label":        label_int,
                    "label_name":   label_name,
                    "source_type":  source_type,
                    "source_domain": d.get("source_domain", ""),
                    "text":         text,
                })

    return pd.DataFrame(rows)


def main():
    if not VFND_DIR.exists():
        sys.exit(
            f"VFND data not found at {VFND_DIR}.\n"
            "Run fetch_data.py first, or manually copy the Dataset/ folder there."
        )

    df = parse_vfnd(VFND_DIR)
    df.to_csv(PROCESSED_CSV, index=False)

    print(f"Parsed {len(df)} articles -> {PROCESSED_CSV}")
    print(f"\nClass balance:")
    print(df["label_name"].value_counts().to_string())
    print(f"\nSource type:")
    print(df["source_type"].value_counts().to_string())
    print(f"\nText length (chars):")
    lengths = df["text"].str.len()
    print(f"  min={lengths.min():.0f}  median={lengths.median():.0f}  "
          f"mean={lengths.mean():.0f}  max={lengths.max():.0f}")

if __name__ == "__main__":
    main()
