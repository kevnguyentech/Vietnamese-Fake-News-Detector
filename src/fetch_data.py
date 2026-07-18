"""
Clones the VFND dataset from GitHub and places it at data/raw/VFND/.

VFND (Vietnamese Fake News Dataset) is an academic dataset of ~260
labeled Vietnamese news articles (Fake / Real) collected 2017-2019.
It lives at: github.com/thanhhocse96/vfnd-vietnamese-fake-news-datasets

If you use this data in any publication or public project, cite:
  Ho Quang Thanh and ninh-pm-se, "thanhhocse96/vfnd-vietnamese-fake-
  news-datasets," Zenodo, Feb. 2019. DOI: 10.5281/zenodo.2578917

Run:
    python src/fetch_data.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

from config import DATA_RAW, VFND_DIR

REPO_URL = "https://github.com/thanhhocse96/vfnd-vietnamese-fake-news-datasets.git"
CLONE_TO  = DATA_RAW / "_vfnd_clone"


def main():
    if VFND_DIR.exists() and any(VFND_DIR.rglob("*.json")):
        count = len(list(VFND_DIR.rglob("*.json")))
        print(f"VFND already present ({count} JSON files at {VFND_DIR}). Nothing to do.")
        print("Delete data/raw/VFND/ and re-run if you want a fresh clone.")
        return

    print(f"Cloning VFND from {REPO_URL} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(CLONE_TO)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"git clone failed:\n{result.stderr}")

    # The repo's data lives in Dataset/
    dataset_src = CLONE_TO / "Dataset"
    if not dataset_src.exists():
        sys.exit(f"Expected Dataset/ inside the clone — not found at {dataset_src}")

    if VFND_DIR.exists():
        shutil.rmtree(VFND_DIR)
    shutil.move(str(dataset_src), str(VFND_DIR))
    shutil.rmtree(CLONE_TO)

    count = len(list(VFND_DIR.rglob("*.json")))
    print(f"Done. {count} JSON files -> {VFND_DIR}")
    print("Next: python src/parse_vfnd.py")


if __name__ == "__main__":
    main()
