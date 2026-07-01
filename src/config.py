"""
Shared constants. One place to change paths, thresholds, and
hyperparameters
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"
OUTPUTS_DIR     = ROOT / "outputs"
VFND_DIR        = DATA_RAW / "VFND"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROCESSED_CSV       = DATA_PROCESSED / "articles.csv"
SEGMENTED_CSV       = DATA_PROCESSED / "articles_segmented.csv"
BASELINE_MODEL_FILE = MODELS_DIR / "tfidf_logreg.pkl"
PHOBERT_DIR         = MODELS_DIR / "phobert"

# Binary labels: Fake=1, Real=0. Misleading dropped (only 2 examples).
LABEL_MAP     = {"Real": 0, "Fake": 1}
LABEL_NAMES   = ["Real", "Fake"]

# Vietnamese sensational-language markers — appear heavily in
# clickbait and misinformation, almost never in sourced journalism.
SENSATIONAL_KEYWORDS = [
    "khẩn cấp", "nóng", "sốc", "kinh hoàng", "chấn động",
    "bí mật", "sự thật", "thực hư", "lộ diện", "phát hiện",
    "cảnh báo", "nguy hiểm", "đặc biệt", "cực kỳ", "tuyệt vời",
    "thần kỳ", "chữa khỏi", "100%", "không thể tin", "chia sẻ ngay",
    "lan truyền", "dân mạng xôn xao", "cư dân mạng", "mạng xã hội",
]

# Phrases typical of properly sourced journalism
CITATION_MARKERS = [
    "theo ", "cho biết", "thông tin", "báo cáo", "nghiên cứu",
    "bộ ", "chính phủ", "cơ quan", "đại diện", "phát ngôn",
    "vnexpress", "tuổi trẻ", "dân trí", "nhân dân", "thanh niên",
]

TOKENIZER             = "underthesea"   # "underthesea" | "pyvi" | "simple"
TFIDF_MAX_FEATURES    = 10_000
TFIDF_NGRAM_RANGE     = (1, 2)
TFIDF_SUBLINEAR_TF    = True
N_FOLDS               = 5
RANDOM_SEED           = 42

LOGREG_MAX_ITER      = 1000
LOGREG_C             = 1.0
LOGREG_CLASS_WEIGHT  = "balanced"

PHOBERT_MODEL_NAME    = "vinai/phobert-base-v2"
PHOBERT_MAX_LEN       = 256
PHOBERT_LR            = 2e-5
PHOBERT_EPOCHS        = 4
PHOBERT_BATCH_SIZE    = 16
PHOBERT_WEIGHT_DECAY  = 0.01
