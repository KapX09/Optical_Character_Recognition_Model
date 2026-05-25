"""
Indian Plate OCR — Training Script
====================================
Wraps fast-plate-ocr CLI with all tunable parameters in one place.
Edit the CONFIG section below, then run:

    python train.py

Requirements:
    pip install fast-plate-ocr scikit-learn
"""

import os
import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ── Dataset ───────────────────────────────────────────────────────────────────
ANNOTATIONS_CSV  = "plates/annotations.csv"   # generated CSV
IMAGES_PREFIX    = "plates/images/"           # prefix added to filenames
TRAIN_CSV        = "train.csv"
VAL_CSV          = "val.csv"
VAL_SPLIT        = 0.15                       # 15% for validation

# ── Plate config ──────────────────────────────────────────────────────────────
PLATE_CONFIG     = "plate_config.yaml"        # preprocessing & vocab config
MODEL_CONFIG     = "model_config.yaml"        # architecture config (optional)

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS           = 100       # increase to 200-300 if accuracy is low
BATCH_SIZE       = 64        # reduce to 32 if OOM error
LEARNING_RATE    = 1e-3      # default; try 5e-4 if loss doesn't converge

# ── Early stopping & LR schedule ─────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 15    # stop if val loss doesn't improve for N epochs
REDUCE_LR_PATIENCE   = 7     # halve LR if val loss stagnates for N epochs

# ── Model architecture ────────────────────────────────────────────────────────
# "cct-xs" = extra small, fastest, ~0.3ms/plate  (use for edge/realtime)
# "cct-s"  = small, more accurate, ~0.6ms/plate  (use if xs accuracy is poor)
MODEL_ARCH       = "cct-s"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "output"   # checkpoints + logs saved here

# ── Backend (keras) ───────────────────────────────────────────────────────────
# Options: "torch" | "tensorflow" | "jax"
# Use torch if you have PyTorch installed, tensorflow otherwise
KERAS_BACKEND    = "torch"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Split dataset into train / val
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset():
    print("\n[1/3] Splitting dataset...")
    df = pd.read_csv(ANNOTATIONS_CSV)

    # fast-plate-ocr expects: filename column = path to image
    df["filename"] = IMAGES_PREFIX + df["filename"]

    # Only keep columns the trainer needs
    df = df[["filename", "plate_text"]]

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    print(f"    Train: {len(train_df)} plates")
    print(f"    Val:   {len(val_df)} plates")
    print(f"    Saved: {TRAIN_CSV}, {VAL_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Validate dataset
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset():
    print("\n[2/3] Validating dataset...")
    cmd = [
        "fast_plate_ocr", "validate-dataset",
        "--annotations", TRAIN_CSV,
        "--config-file", PLATE_CONFIG,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nDataset validation failed. Fix errors above before training.")
        sys.exit(1)
    print("✓ Dataset valid.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Train
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("\n[3/3] Training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = os.environ.copy()
    env["KERAS_BACKEND"] = KERAS_BACKEND

    cmd = [
        "fast_plate_ocr", "train",
        "--annotations",             TRAIN_CSV,
        "--val-annotations",         VAL_CSV,
        "--config-file",             PLATE_CONFIG,
        "--batch-size",              str(BATCH_SIZE),
        "--epochs",                  str(EPOCHS),
        "--learning-rate",           str(LEARNING_RATE),
        "--early-stopping-patience", str(EARLY_STOP_PATIENCE),
        "--reduce-lr-patience",      str(REDUCE_LR_PATIENCE),
        "--output-dir",              OUTPUT_DIR,
    ]

    print("    Command:", " ".join(cmd))
    print(f"    Backend: {KERAS_BACKEND}")
    print(f"    Arch:    {MODEL_ARCH}")
    print(f"    Epochs:  {EPOCHS} (early stop patience={EARLY_STOP_PATIENCE})")
    print(f"    Batch:   {BATCH_SIZE}")
    print(f"    LR:      {LEARNING_RATE}\n")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print("\nTraining failed. Check errors above.")
        sys.exit(1)

    print(f"\n✓ Training complete. Model saved to: {OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Indian Plate OCR — Training")
    print("=" * 60)

    # Check plate_config exists
    if not os.path.exists(PLATE_CONFIG):
        print(f"Missing: {PLATE_CONFIG} — make sure it's in this folder.")
        sys.exit(1)

    # Check annotations CSV exists
    if not os.path.exists(ANNOTATIONS_CSV):
        print(f"Missing: {ANNOTATIONS_CSV} — run generate_indian_plates.py first.")
        sys.exit(1)

    split_dataset()
    validate_dataset()
    train()