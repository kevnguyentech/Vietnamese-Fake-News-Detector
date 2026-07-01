# Vietnamese Fake News Detector

Misinformation spreads fast on Vietnamese Facebook and Zalo, and every
existing fake-news detector is built for English. This project trains a
classifier on **real labeled Vietnamese news articles** (the VFND academic
dataset) to detect fake news in Vietnamese text - with a plain-language
explanation of exactly which words drove each prediction.

It's also a deliberate teaching example of how to handle a small NLP
dataset correctly: classical baseline before any transformer, stratified
k-fold instead of a single lucky split, domain-specific feature engineering,
and an error analysis that reads the actual wrong predictions instead of just
reporting aggregate numbers.

---

## Quick start

```bash
pip install -r requirements.txt
cd src

python parse_vfnd.py          # parse 272 labeled JSON files -> clean CSV
python preprocess.py          # Vietnamese segmentation + handcrafted features
python baseline.py            # TF-IDF + LogReg, 5-fold CV -> saves model
python evaluate.py            # confusion matrix + top features chart
python predict.py "SỐC! Bác sĩ tiết lộ bí mật chữa khỏi ung thư 100% bằng lá cây rừng!!!"
```

---

## The data: VFND

**272 labeled Vietnamese news articles** collected 2017–2019:
144 Fake, 128 Real (plus 2 Misleading, dropped — too few to form a class).
Each article comes from the VFND dataset by Ho Quang Thanh and ninh-pm-se,
archived at Zenodo with DOI 10.5281/zenodo.2578917. If you use this data
in any publication, cite the original authors.

The dataset splits into `Article_Contents` (formal news, ~260 articles) and
`Social_Contents` (Facebook posts, 12 articles). Both are included.
`fetch_data.py` clones the dataset automatically from GitHub; it's already
included in `data/raw/VFND/` in this repo.

---

## Why a classical baseline before PhoBERT

With only 272 examples, a 135M-parameter transformer has enormous capacity
relative to the data. TF-IDF + Logistic Regression:

- Trains in under 2 seconds
- Achieves **83.0% macro F1** (mean across 5 folds)
- Can explain every prediction in plain terms (coefficient weights)
- Gives PhoBERT a number to beat


---

## Results

**TF-IDF + LogReg baseline — 5-fold stratified CV:**

| Fold | Accuracy | Macro F1 | Real F1 | Fake F1 |
|------|----------|----------|---------|---------|
| 1    | 0.891    | 0.890    | 0.880   | 0.900   |
| 2    | 0.727    | 0.724    | 0.754   | 0.694   |
| 3    | 0.852    | 0.852    | 0.857   | 0.846   |
| 4    | 0.870    | 0.870    | 0.863   | 0.877   |
| 5    | 0.815    | 0.812    | 0.792   | 0.833   |
| **Mean** | **0.831 ± 0.058** | **0.830 ± 0.059** | | |

The ±0.059 std is the honest number - with 272 examples, a single lucky
80/20 split could report anywhere from 0.72 to 0.89. 5-fold CV shows you
the real range. Fold 2 being lower is not a bug; that particular held-out
set happened to contain harder borderline articles.

**What the model learned** (`outputs/top_features.png`):

- **→ FAKE**: `exclamation_count` by a wide margin, then `sensational_count`,
  `question_count`. Vietnamese clickbait is heavily punctuation-marked.
  Specific celebrity gossip terms (gái, xinh, nóng_bỏng, mc) cluster in
  the Fake side — a large chunk of the fake articles are entertainment gossip.

- **→ REAL**: `citation_count` (legitimate journalism cites sources),
  `digit_ratio` (real news has dates, statistics, case counts),
  `char_count` (real articles are longer — more reporting, less shouting).


---

## Why Vietnamese word segmentation matters

Vietnamese is a tonal multi-syllabic language. "học sinh" (student) is one
semantic unit written as two space-separated syllables. A whitespace tokenizer
splits it into "học" and "sinh" - two meaningless fragments. With proper
segmentation (underthesea), it becomes "học_sinh" - one meaningful TF-IDF
token. PhoBERT was pretrained on pre-segmented text and performs noticeably
worse without it, because its vocabulary was built from segmented input.

---

## PhoBERT fine-tuning

`train_phobert.py` fine-tunes `vinai/phobert-base-v2` on the same 5-fold
splits so results are directly comparable to the baseline. It requires
internet access (model weights from HuggingFace, ~550MB) and is recommended
to run with a GPU (Google Colab free tier is enough).

```bash
pip install transformers torch datasets
python src/train_phobert.py
```

Then use it in predict.py:
```bash
python src/predict.py --model phobert "your Vietnamese text here"
```

Expected improvement over baseline: 3–8 points macro F1 on this dataset.
If it's less than that, the dataset is the bottleneck, not the model choice.

---

## Error analysis

`outputs/error_analysis.csv` lists the 10 most confidently wrong predictions:

- **Fake predicted as Real**: Several entertainment gossip articles written in
  a formal register (full sentences, no punctuation inflation) that the model
  has no way to flag without knowing the underlying facts are false.
- **Real predicted as Fake**: Legitimate news articles with unusually
  emotional headlines ("Kinh hoàng bé gái bị chó nhà tấn công") that read
  like clickbait but aren't. The model can only see surface patterns,
  not whether the reported event actually happened.

Both failure modes point to the same underlying limit: the model detects
**style**, not **truth**. A well-written lie fools it. A dramatically-titled
true story fools it the other way. This is not fixable by adding more features
or switching to PhoBERT — it's fundamental to text-only classification without
external fact-checking.

---

## Project structure

```
vietnamese-fake-news-detector/
├── data/
│   ├── raw/VFND/              # 272 labeled JSON files (Fake/ + Real/)
│   └── processed/
│       ├── articles.csv       # parsed: id, label, text, source_type
│       └── articles_segmented.csv  # + cleaned text, segmented, features
├── src/
│   ├── config.py              # all paths + hyperparameters in one place
│   ├── fetch_data.py          # git clone VFND from GitHub
│   ├── parse_vfnd.py          # JSON folder → flat CSV
│   ├── preprocess.py          # cleaning + underthesea segmentation + features
│   ├── baseline.py            # TF-IDF + LogReg, 5-fold CV, saves model
│   ├── train_phobert.py       # PhoBERT fine-tuning (needs GPU + internet)
│   ├── evaluate.py            # confusion matrix + top features + error analysis
│   └── predict.py             # CLI tool — run this on any Vietnamese text
├── models/
│   ├── tfidf_logreg.pkl       # saved baseline (loaded by predict.py)
│   └── phobert/final/         # saved PhoBERT (after train_phobert.py)
├── outputs/
│   ├── confusion_matrix.png
│   ├── top_features.png
│   └── error_analysis.csv
└── requirements.txt
```

---

## Limitations

The model detects style, not truth. Three specific limits worth knowing:

**Dataset size**: 272 articles is small. The ±0.059 std across folds is real
variance, not noise to ignore. More data would matter more than a better model.

**2017–2019 vintage**: The dataset was collected during those years. Fake-news
patterns evolve; a model trained on 2018 Vietnamese tabloid patterns may miss
2024 AI-generated misinformation that mimics formal journalistic style.

**Entertainment bias**: A large fraction of the Fake articles are celebrity
gossip ("gái xinh nóng bỏng", "MC Lại Văn Sâm"), which makes those features
heavily weighted. The model may underperform on political or health
misinformation that doesn't use the same vocabulary.

