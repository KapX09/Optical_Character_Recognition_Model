"""
Indian License Plate Synthetic Dataset Generator
Output: cropped plate images + annotations.csv (fast-plate-ocr compatible)

Indian plate format: XX00XX0000
  - State code (2 letters): MH, GJ, DL, KA, TN, UP, RJ, WB, AP, TS, MP, HR, PB, etc.
  - District number (2 digits): 01-99
  - Series (2 letters): AA-ZZ
  - Number (4 digits): 0001-9999

Plate types:
  - White bg, black text (private vehicles)
  - Yellow bg, black text (commercial vehicles)
  - Green bg, white text (electric vehicles)
"""

import os
import csv
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Config 
NUM_IMAGES   = 5000
OUTPUT_DIR   = "indian_plates_dataset"
IMG_W, IMG_H = 200, 60   # fast-plate-ocr default input size
FONT_PATH    = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
FONT_SIZE    = 36

STATE_CODES = [
    "MH","GJ","DL","KA","TN","UP","RJ","WB","AP","TS",
    "MP","HR","PB","BR","OR","KL","AS","JH","UK","HP",
    "GA","MN","MZ","NL","SK","TR","AR","ML","CG","JK",
]

SERIES_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O (ambiguous)

PLATE_TYPES = [
    {"bg": (255,255,255), "fg": (0,0,0),   "weight": 60},  # white - private
    {"bg": (255,220,0),   "fg": (0,0,0),   "weight": 30},  # yellow - commercial
    {"bg": (0,130,0),     "fg": (255,255,255), "weight": 10}, # green - EV
]

#  Helpers 

def weighted_choice(choices):
    weights = [c["weight"] for c in choices]
    total   = sum(weights)
    r       = random.uniform(0, total)
    upto    = 0
    for c in choices:
        upto += c["weight"]
        if r <= upto:
            return c
    return choices[-1]

def random_plate_text():
    state   = random.choice(STATE_CODES)
    dist    = f"{random.randint(1,99):02d}"
    series  = random.choice(SERIES_LETTERS) + random.choice(SERIES_LETTERS)
    number  = f"{random.randint(1,9999):04d}"
    return f"{state}{dist}{series}{number}"   # e.g. MH04AB1234

def add_noise(img, intensity=15):
    arr   = np.array(img).astype(np.int16)
    noise = np.random.randint(-intensity, intensity, arr.shape, dtype=np.int16)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_shadow(img):
    """Simulate uneven lighting by darkening a gradient region."""
    arr    = np.array(img).astype(np.float32)
    w      = arr.shape[1]
    start  = random.randint(0, w // 2)
    end    = random.randint(start + 10, w)
    factor = random.uniform(0.55, 0.85)
    arr[:, start:end, :3] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def rotate_image(img, max_deg=3.5):
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, expand=False, fillcolor=img.getpixel((0, 0)))

def perspective_warp(img, strength=0.04):
    """Slight horizontal keystone."""
    w, h = img.size
    shift = int(w * strength * random.choice([-1, 1]))
    src   = [(0,0),(w,0),(w,h),(0,h)]
    dst   = [(max(0,shift),0),(w-max(0,-shift),0),
             (w-max(0,shift),h),(max(0,-shift),h)]
    coeffs = find_coeffs(dst, src)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix += [[p2[0],p2[1],1,0,0,0,-p1[0]*p2[0],-p1[0]*p2[1]],
                   [0,0,0,p2[0],p2[1],1,-p1[1]*p2[0],-p1[1]*p2[1]]]
    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pa).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.array(res).reshape(8)

def add_dirt(img, spots=6):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(0, spots)):
        x, y = random.randint(0,w), random.randint(0,h)
        r    = random.randint(1, 4)
        col  = (random.randint(80,160),)*3
        draw.ellipse([x-r, y-r, x+r, y+r], fill=col)
    return img

#  Plate renderer 

def render_plate(text, plate_type, font):
    img  = Image.new("RGB", (IMG_W, IMG_H), plate_type["bg"])
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([2,2,IMG_W-3,IMG_H-3], outline=plate_type["fg"], width=2)

    # Center text
    bbox = draw.textbbox((0,0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = (IMG_W - tw) // 2 - bbox[0]
    y    = (IMG_H - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=plate_type["fg"])

    return img

def augment(img, plate_type):
    # Always: noise
    img = add_noise(img, intensity=random.randint(5, 20))

    # 60%: shadow
    if random.random() < 0.6:
        img = add_shadow(img)

    # 40%: slight blur
    if random.random() < 0.4:
        img = add_blur(img, radius=random.uniform(0.3, 0.9))

    # 70%: rotation
    if random.random() < 0.7:
        img = rotate_image(img, max_deg=3.5)

    # 50%: perspective
    if random.random() < 0.5:
        img = perspective_warp(img, strength=random.uniform(0.01, 0.04))

    # 30%: dirt
    if random.random() < 0.3:
        img = add_dirt(img, spots=random.randint(2, 8))

    return img

#  Main 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    rows = []
    seen = set()

    print(f"Generating {NUM_IMAGES} plates...")

    i = 0
    while i < NUM_IMAGES:
        text  = random_plate_text()
        ptype = weighted_choice(PLATE_TYPES)

        img   = render_plate(text, ptype, font)
        img   = augment(img, ptype)

        # unique filename even if same plate text appears twice
        fname = f"{text}_{i:05d}.jpg"
        img.save(os.path.join(img_dir, fname), quality=92)
        rows.append({"filename": fname, "plate": text})

        i += 1
        if i % 500 == 0:
            print(f"  {i}/{NUM_IMAGES} done")

    # Write annotations.csv
    csv_path = os.path.join(OUTPUT_DIR, "annotations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","plate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {NUM_IMAGES} images saved to '{OUTPUT_DIR}/'")
    print(f"CSV: {csv_path}")
    print(f"\nfast-plate-ocr dataset structure:")
    print(f"  {OUTPUT_DIR}/images/*.jpg")
    print(f"  {OUTPUT_DIR}/annotations.csv  [filename, plate]")

if __name__ == "__main__":
    main()
