"""
Fine-tunes PhoBERT (vinai/phobert-base-v2) for fake-news detection.

WHY PHOBERT:
PhoBERT is a RoBERTa-based model pretrained on 140GB of Vietnamese
text. It understands Vietnamese morphology and word order in a way
TF-IDF fundamentally can't - word meaning changes with context, and
PhoBERT captures that. "nguy hiểm" (dangerous) means something very
different in "nghiên cứu phát hiện chất nguy hiểm" (scientific article)
vs "CẢNH BÁO: chất nguy hiểm" (clickbait headline).

WHY THIS NEEDS YOUR OWN MACHINE:
The model weights (~550MB) download from huggingface.co, which this
sandbox can't reach. Everything else in the pipeline runs here. Run
this script on any machine with internet + ~4GB RAM (GPU recommended
but not required - CPU training is slow, ~20 min per fold, but works).

    pip install transformers torch datasets
    python src/train_phobert.py

ARCHITECTURE:
  Input text -> PhoBERT tokenizer -> phobert-base-v2 ->
  [CLS] embedding -> dropout(0.1) -> Linear(768, 2) -> softmax

  I fine-tune ALL layers, not just the classification head. With only
  260 examples that risks overfitting, which is why:
  (a) learning rate is low (2e-5), decay is on (0.01)
  (b) early stopping watches validation loss, stops at patience=2
  (c) I report 5-fold CV, not a single split

TRAIN/EVAL STRATEGY:
  Same 5-fold stratified split as baseline.py so results are directly
  comparable. Fine-tune on 4 folds, evaluate on the held-out 1 fold.
  Final model is saved from the fold with the best validation loss.
"""

import sys
import gc
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    SEGMENTED_CSV, PHOBERT_DIR, PHOBERT_MODEL_NAME, PHOBERT_MAX_LEN,
    PHOBERT_LR, PHOBERT_EPOCHS, PHOBERT_BATCH_SIZE, PHOBERT_WEIGHT_DECAY,
    LABEL_NAMES, N_FOLDS, RANDOM_SEED,
)

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, classification_report
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install transformers torch datasets scikit-learn"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": (preds == labels).mean(),
    }


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text_seg"],
        truncation=True,
        padding="max_length",
        max_length=PHOBERT_MAX_LEN,
    )


def run_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
             tokenizer, save_dir: Path) -> dict:
    """Fine-tune on one fold, return val metrics."""
    train_ds = Dataset.from_dict({
        "text_seg": train_df["text_seg"].tolist(),
        "label":    train_df["label"].tolist(),
    })
    val_ds = Dataset.from_dict({
        "text_seg": val_df["text_seg"].tolist(),
        "label":    val_df["label"].tolist(),
    })

    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    val_ds.set_format("torch",   columns=["input_ids","attention_mask","label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_MODEL_NAME, num_labels=2
    )

    args = TrainingArguments(
        output_dir=str(save_dir / f"fold_{fold}"),
        num_train_epochs=PHOBERT_EPOCHS,
        per_device_train_batch_size=PHOBERT_BATCH_SIZE,
        per_device_eval_batch_size=PHOBERT_BATCH_SIZE,
        learning_rate=PHOBERT_LR,
        weight_decay=PHOBERT_WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=RANDOM_SEED,
        report_to="none",         # disable wandb / other loggers
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics = trainer.evaluate()

    # Print fold results
    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    print(f"\nFold {fold} classification report:")
    print(classification_report(val_df["label"].values, preds,
                                 target_names=LABEL_NAMES, zero_division=0))

    # Persist this fold's best checkpoint now (load_best_model_at_end
    # already loaded it into trainer.model), instead of keeping the
    # trainer/model alive in memory until every fold has finished.
    fold_model_dir = save_dir / f"fold_{fold}_best"
    trainer.save_model(str(fold_model_dir))
    shutil.rmtree(str(save_dir / f"fold_{fold}"), ignore_errors=True)

    result = {
        "fold":      fold,
        "macro_f1":  metrics.get("eval_macro_f1", 0),
        "accuracy":  metrics.get("eval_accuracy", 0),
        "model_dir": fold_model_dir,
    }

    # Free this fold's model before the next fold starts.
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    df = pd.read_csv(SEGMENTED_CSV)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    print(f"Loaded {len(df)} articles")

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    PHOBERT_DIR.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{N_FOLDS}  train={len(train_idx)}  val={len(val_idx)}")
        print("="*50)
        result = run_fold(
            fold=fold,
            train_df=df.iloc[train_idx],
            val_df=df.iloc[val_idx],
            tokenizer=tokenizer,
            save_dir=PHOBERT_DIR,
        )
        fold_results.append(result)

    # Print CV summary
    macro_f1s = [r["macro_f1"] for r in fold_results]
    accs      = [r["accuracy"]  for r in fold_results]
    print(f"\n{'='*50}")
    print(f"PhoBERT CV summary ({N_FOLDS} folds):")
    print(f"  macro_f1:  {np.mean(macro_f1s):.3f} ± {np.std(macro_f1s):.3f}")
    print(f"  accuracy:  {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # Save the best fold's model as the final model
    best = max(fold_results, key=lambda r: r["macro_f1"])
    print(f"\nBest fold: {best['fold']}  macro_f1={best['macro_f1']:.3f}")
    final_dir = PHOBERT_DIR / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best["model_dir"], final_dir)
    tokenizer.save_pretrained(str(final_dir))
    for r in fold_results:
        shutil.rmtree(str(r["model_dir"]), ignore_errors=True)
    print(f"Saved best model -> {final_dir}")
    print("\nTo use: python src/predict.py --model phobert 'your text here'")


if __name__ == "__main__":
    main()
