"""
Indian Plate OCR — Evaluate & Inference
========================================
1. Eval mode   — full accuracy on val.csv with metrics + sample grid saved as image
2. Infer mode  — run on single image or folder, show results visually

Usage:
    python eval/evaluate.py --mode eval
    python eval/evaluate.py --mode eval --limit 40
    python eval/evaluate.py --mode eval --save results.csv

    python eval/evaluate.py --mode infer --input path/to/plate.jpg
    python eval/evaluate.py --mode infer --input path/to/folder/
    python eval/evaluate.py --mode infer --input path/to/folder/ --save results.csv
"""

import os
import sys
import argparse
import time
import csv
import pandas as pd
import numpy as np
import cv2
from fast_plate_ocr import LicensePlateRecognizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#  CONFIG
MODEL_PATH   = os.path.join(ROOT, "model", "output", "timestamp", "best.onnx")
PLATE_CONFIG = os.path.join(ROOT, "config", "plate_config.yaml")
VAL_CSV      = os.path.join(ROOT, "dataset", "val.csv")
BATCH_SIZE   = 32
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}

# Visual display settings for SAMPLE grids only (not all 5000)
DISPLAY_W      = 280
DISPLAY_H      = 80
LABEL_H        = 30
GRID_COLS      = 4
SHOW_N_ERRORS  = 20    # sample wrong plates saved to image
SHOW_N_CORRECT = 8     # sample correct plates saved to image

#  LOAD MODEL
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"Loading model: {MODEL_PATH}")
    model = LicensePlateRecognizer(
        hub_ocr_model=None,
        onnx_model_path=MODEL_PATH,
        plate_config_path=PLATE_CONFIG
    )
    print("✓ Model loaded\n")
    return model

#  IMAGE PREP
def prepare(img):
    """Single-channel grayscale — model expects 1 channel."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  VISUAL HELPERS
def make_plate_tile(img, label_top, label_bot=None, correct=None):
    tile_img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    if correct is True:    border_color = (40, 180, 40)
    elif correct is False: border_color = (40, 40, 220)
    else:                  border_color = (160, 160, 160)
    cv2.rectangle(tile_img, (0,0), (DISPLAY_W-1, DISPLAY_H-1), border_color, 2)

    rows = 1 if label_bot is None else 2
    label_area = np.ones((LABEL_H * rows, DISPLAY_W, 3), dtype=np.uint8) * 30
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1

    if label_bot is None:
        (tw,_),_ = cv2.getTextSize(label_top, font, fs, th)
        cv2.putText(label_area, label_top, (max(0,(DISPLAY_W-tw)//2), 20), font, fs, (220,220,220), th)
    else:
        (tw,_),_ = cv2.getTextSize(label_top, font, fs, th)
        cv2.putText(label_area, label_top, (max(0,(DISPLAY_W-tw)//2), 18), font, fs, (200,200,200), th)
        pred_color = (40,200,40) if correct else (80,80,255)
        (tw2,_),_ = cv2.getTextSize(label_bot, font, fs, th)
        cv2.putText(label_area, label_bot, (max(0,(DISPLAY_W-tw2)//2), 42), font, fs, pred_color, th)
    return np.vstack([tile_img, label_area])


def make_grid(tiles, cols=GRID_COLS):
    if not tiles: return None
    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i:i+cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.ones_like(tiles[0]) * 20)
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def add_header(grid, text, color=(220,220,220)):
    if grid is None: return None
    h = np.ones((36, grid.shape[1], 3), dtype=np.uint8) * 45
    cv2.putText(h, text, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return np.vstack([h, grid])

#  METRICS HELPERS
def char_error_rate(pred, truth):
    """CER = edit distance / length of truth."""
    import difflib
    matcher = difflib.SequenceMatcher(None, pred, truth)
    edits = sum(max(len(pred[i1:i2]), len(truth[j1:j2]))
                for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != 'equal')
    return edits / max(len(truth), 1)

#  EVALUATION
def evaluate(model, save_path=None, limit=None):
    if not os.path.exists(VAL_CSV):
        print(f"Missing: {VAL_CSV}")
        sys.exit(1)

    df = pd.read_csv(VAL_CSV)
    if limit:
        df = df.head(limit)

    total         = len(df)
    full_correct  = 0
    char_correct  = 0
    char_total    = 0
    cer_total     = 0.0
    len_correct   = 0
    dataset_dir   = os.path.join(ROOT, "dataset")
    correct_tiles = []
    error_tiles   = []
    save_rows     = []

    # Per state-code breakdown
    state_stats = {}

    print(f"Evaluating on {total} plates...\n")
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch    = df.iloc[i:i+BATCH_SIZE]
        images   = []
        img_list = []

        for _, row in batch.iterrows():
            full_path = os.path.join(dataset_dir, row["image_path"])
            img = cv2.imread(full_path)
            if img is None:
                img = np.zeros((70, 140, 3), dtype=np.uint8)
            img_list.append(img)
            images.append(prepare(img))

        predictions = model.run(images)

        for j, (_, row) in enumerate(batch.iterrows()):
            pred  = predictions[j].plate.strip().upper()
            truth = str(row["plate_text"]).strip().upper()
            ok    = pred == truth

            # Full plate
            if ok:
                full_correct += 1
                if len(correct_tiles) < SHOW_N_CORRECT:
                    correct_tiles.append(make_plate_tile(img_list[j], f"GT: {truth}", f"OK: {pred}", correct=True))
            else:
                if len(error_tiles) < SHOW_N_ERRORS:
                    error_tiles.append(make_plate_tile(img_list[j], f"GT: {truth}", f"PR: {pred}", correct=False))

            # Per-char accuracy
            for c_pred, c_true in zip(pred, truth):
                char_total += 1
                if c_pred == c_true:
                    char_correct += 1
            char_total += abs(len(pred) - len(truth))

            # Length accuracy
            if len(pred) == len(truth):
                len_correct += 1

            # CER
            cer_total += char_error_rate(pred, truth)

            # Per state code (first 2 chars)
            state = truth[:2] if len(truth) >= 2 else "??"
            if state not in state_stats:
                state_stats[state] = {"correct": 0, "total": 0}
            state_stats[state]["total"] += 1
            if ok:
                state_stats[state]["correct"] += 1

            if save_path:
                save_rows.append({
                    "filename": row["image_path"],
                    "truth": truth,
                    "pred": pred,
                    "correct": ok,
                    "cer": round(char_error_rate(pred, truth), 4)
                })

        print(f"  {min(i+BATCH_SIZE, total)}/{total}", end="\r")

    elapsed   = time.time() - start
    plate_acc = full_correct / total * 100
    char_acc  = char_correct / char_total * 100 if char_total > 0 else 0
    len_acc   = len_correct / total * 100
    avg_cer   = cer_total / total * 100

    # Print metrics
    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Total plates       : {total}")
    print(f"  Full plate acc     : {plate_acc:.2f}%   ← entire plate exact match")
    print(f"  Per-char acc       : {char_acc:.2f}%   ← individual characters")
    print(f"  Length acc         : {len_acc:.2f}%   ← correct plate length")
    print(f"  Avg CER            : {avg_cer:.2f}%   ← char error rate (lower=better)")
    print(f"  Correct plates     : {full_correct}/{total}")
    print(f"  Wrong plates       : {total - full_correct}/{total}")
    print(f"  Time               : {elapsed:.1f}s")
    print(f"  Avg per plate      : {elapsed/total*1000:.1f}ms")
    print("=" * 52)

    if plate_acc >= 90:   grade = "Excellent"
    elif plate_acc >= 75: grade = "Good — consider more data or epochs"
    elif plate_acc >= 50: grade = "Fair — retrain with real plate crops"
    else:                 grade = "Poor — check font, config, or data"
    print(f"\n  Grade: {grade}")

    # Per state breakdown (top 5 worst)
    print(f"\n  Per state accuracy (worst 5):")
    print(f"  {'State':<8} {'Acc':<10} {'Correct/Total'}")
    print(f"  {'-'*35}")
    sorted_states = sorted(state_stats.items(), key=lambda x: x[1]["correct"]/x[1]["total"])
    for state, s in sorted_states[:5]:
        acc = s["correct"] / s["total"] * 100
        print(f"  {state:<8} {acc:<10.1f} {s['correct']}/{s['total']}")

    print()

    # Save CSV
    if save_path:
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename","truth","pred","correct","cer"])
            writer.writeheader()
            writer.writerows(save_rows)
        print(f"  Results saved: {save_path}")

    # Save sample grids as images
    if correct_tiles:
        g = make_grid(correct_tiles)
        g = add_header(g, f"Correct samples ({len(correct_tiles)})", (60,200,60))
        cv2.imwrite("eval_correct.jpg", g)
        print("  Saved: eval_correct.jpg")

    if error_tiles:
        g = make_grid(error_tiles)
        g = add_header(g, f"Wrong samples ({len(error_tiles)})  GT=ground truth  PR=predicted", (80,80,220))
        cv2.imwrite("eval_errors.jpg", g)
        print("  Saved: eval_errors.jpg")

#  INFERENCE
def infer(model, input_path, save_path=None, limit=None):
    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if not image_paths:
            print(f"No images found in: {input_path}")
            sys.exit(1)
    else:
        print(f"Not a file or folder: {input_path}")
        sys.exit(1)

    if limit:
        image_paths = image_paths[:limit]

    print(f"Running inference on {len(image_paths)} image(s)...\n")

    results = []
    tiles   = []
    start   = time.time()

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        images, imgs_raw = [], []

        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((70, 140, 3), dtype=np.uint8)
            imgs_raw.append(img)
            images.append(prepare(img))

        predictions = model.run(images)

        for path, pred, img in zip(batch_paths, predictions, imgs_raw):
            plate_text = pred.plate.strip().upper()
            fname      = os.path.basename(path)
            results.append((fname, plate_text))
            print(f"  {fname:<35} → {plate_text}")
            tiles.append(make_plate_tile(img, plate_text, correct=None))

    elapsed = time.time() - start
    print(f"\n  Done. {len(results)} plates in {elapsed:.2f}s")
    print(f"  Avg: {elapsed/len(results)*1000:.1f}ms per plate")

    if save_path:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "plate_text"])
            writer.writerows(results)
        print(f"  Results saved: {save_path}")

    if tiles:
        g = make_grid(tiles)
        g = add_header(g, f"Inference — {len(results)} plates")
        cv2.imwrite("infer_results.jpg", g)
        print("  Saved: infer_results.jpg")


#  MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Evaluate & Infer")
    parser.add_argument("--mode",  choices=["eval", "infer"], required=True)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--save",  type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model()

    if args.mode == "eval":
        evaluate(model, save_path=args.save, limit=args.limit)
    elif args.mode == "infer":
        if not args.input:
            print("--input required for infer mode")
            sys.exit(1)
        infer(model, args.input, save_path=args.save, limit=args.limit)
