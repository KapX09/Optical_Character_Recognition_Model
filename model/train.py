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
import shutil

CLI = (
    shutil.which("fast-plate-ocr")
    or r"E:\A_code\set_up_az\set_C\py10\venv\Scripts\fast-plate-ocr.exe"
)

# Always resolve paths relative to project root (one level up from model/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



#  CONFIG
#  Dataset 
ANNOTATIONS_CSV = os.path.join(ROOT, "dataset/plates/annotations.csv")   # generated CSV
# IMAGES_PREFIX   = os.path.join(ROOT, "dataset/plates/images/")           # prefix added to filenames
TRAIN_CSV       = os.path.join(ROOT, "dataset/train.csv")
VAL_CSV         = os.path.join(ROOT, "dataset/val.csv")
VAL_SPLIT       = 0.15                       # 15% for validation

#  Plate config 
PLATE_CONFIG     = os.path.join(ROOT,"config/plate_config.yaml")        # preprocessing & vocab config
MODEL_CONFIG     = os.path.join(ROOT,"config/model_config.yaml")        # architecture config (optional)

#  Training hyperparameters 
EPOCHS           = 100       # increase to 200-300 if accuracy is low
BATCH_SIZE       = 16        # reduce to 32 if OOM error
LEARNING_RATE    = 1e-3      # default; try 5e-4 if loss doesn't converge

#  Early stopping & LR schedule 
EARLY_STOP_PATIENCE  = 15    # stop if val loss doesn't improve for N epochs
REDUCE_LR_PATIENCE   = 7     # halve LR if val loss stagnates for N epochs

#  Model architecture 
# "cct-xs" = extra small, fastest, ~0.3ms/plate  (use for edge/realtime)
# "cct-s"  = small, more accurate, ~0.6ms/plate  (use if xs accuracy is poor)
MODEL_ARCH       = "cct-s"

#  Output 
OUTPUT_DIR       = os.path.join(ROOT,"model/output")   # checkpoints + logs saved here

#  Backend (keras) 
# Options: "torch" | "tensorflow" | "jax"
# Use torch if you have PyTorch installed, tensorflow otherwise
KERAS_BACKEND    = "torch"

#  Split dataset into train / val
def split_dataset():
    print("\n[1/3] Splitting dataset...")
    df = pd.read_csv(ANNOTATIONS_CSV)

    # fast-plate-ocr expects: filename column = path to image
    # df["filename"] = IMAGES_PREFIX + df["filename"]

    # Only keep columns the trainer needs
    df["image_path"] = df["filename"].apply(
        lambda f: f"plates/images/{f}"
    )
    df = df[["image_path", "plate_text"]]

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

# Validate dataset

def validate_dataset():
    print("\n[2/3] Validating dataset...")
    cmd = [
        CLI, "validate-dataset",
        "--annotations-file", TRAIN_CSV,
        "--plate-config-file", PLATE_CONFIG,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nDataset validation failed. Fix errors above before training.")
        sys.exit(1)
    print("✓ Dataset valid.")


# Train
def train():
    print("\n[3/3] Training...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = os.environ.copy()
    env["KERAS_BACKEND"] = KERAS_BACKEND
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    cmd = [
        CLI, "train",
        "--annotations",              TRAIN_CSV,
        "--val-annotations",          VAL_CSV,
        "--plate-config-file",        PLATE_CONFIG,
        "--model-config-file",        MODEL_CONFIG,
        "--batch-size",               str(BATCH_SIZE),
        "--epochs",                   str(EPOCHS),
        "--lr",                       str(LEARNING_RATE),
        "--early-stopping-patience",  str(EARLY_STOP_PATIENCE),
        "--output-dir",               OUTPUT_DIR,
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


# MAIN
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
