"""
Indian License Plate Synthetic Dataset Generator
=================================================
Generates HSRP-accurate Indian plates (primary) + foreign plates (minority).

Output:
    plates/
        images/          <- JPGs
        annotations.csv  <- filename, plate_text, plate_type

Usage:
    python generate_indian_plates.py
    python generate_indian_plates.py --num 10000 --out my_dataset
"""

import os
import csv
import random
import argparse
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# CONFIG
NUM_IMAGES      = 5000
OUT_DIR         = "plates"

# Distribution
INDIA_RATIO     = 0.85   # 85% Indian plates
FOREIGN_RATIO   = 0.15   # 15% foreign / old-style

# Indian sub-distribution
INDIA_HSRP_RATIO   = 0.75   # of Indian plates: modern HSRP
INDIA_OLD_RATIO    = 0.25   # of Indian plates: old non-standard fonts/styles

# Plate dimensions (4-wheeler, scaled down for dataset)
PLATE_W, PLATE_H = 400, 120

#  FONTs
# Put CharlesWright-Bold.otf in the same folder as this script
MAIN_FONT_PATH   = "CharlesWright-Bold.otf"          # HSRP plates (required)

# Windows built-in fonts for variety (old Indian + foreign plates)
MONO_FONT_PATH   = "C:/Windows/Fonts/cour.ttf"       # Courier New
ROBOTO_FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"    # Arial Bold
CONDENSED_PATH   = "C:/Windows/Fonts/arialbd.ttf"    # Arial Bold
SERIF_FONT_PATH  = "C:/Windows/Fonts/timesbd.ttf"    # Times New Roman Bold
SANS_FONT_PATH   = "C:/Windows/Fonts/arialbd.ttf"    # Arial Bold
FREE_SANS_PATH   = "C:/Windows/Fonts/verdanab.ttf"   # Verdana Bold
def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

#  STATE CODES 
STATE_CODES = [
    "MH","GJ","DL","KA","TN","RJ","UP","WB","MP","AP",
    "TS","KL","HR","PB","BR","OR","JH","UK","HP","GA",
    "CH","JK","AS","MN","ML","NL","TR","SK","AR","MZ",
    "DN","DD","LD","PY","AN"
]

# District numbers per state (simplified: 01-99)
def rand_district():
    return str(random.randint(1, 99)).zfill(2)

def rand_letters():
    # Avoid ambiguous chars O,I,Q
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return random.choice(chars) + random.choice(chars)

def rand_digits():
    return str(random.randint(1000, 9999))

def generate_plate_text():
    state = random.choice(STATE_CODES)
    return f"{state}{rand_district()}{rand_letters()}{rand_digits()}"

#  COLORS
COLORS = {
    "white_bg":   (245, 245, 240),
    "yellow_bg":  (255, 215, 0),
    "green_bg":   (0, 160, 80),
    "black_text": (10, 10, 10),
    "white_text": (245, 245, 240),
    "blue_india": (0, 80, 180, 60),   # semi-transparent for INDIA watermark
    "border":     (30, 30, 30),
}

def plate_type_and_colors():
    """Returns (bg_color, text_color, plate_type_label, has_green_border)"""
    r = random.random()
    if r < 0.60:
        return COLORS["white_bg"],  COLORS["black_text"], "private",    False
    elif r < 0.90:
        return COLORS["yellow_bg"], COLORS["black_text"], "commercial", False
    else:
        return COLORS["white_bg"],  COLORS["black_text"], "ev",         True

#  REFLECTIVE TEXTURE
def add_reflective_texture(img):
    """Adds subtle shimmer to simulate reflective aluminium."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3, arr.shape[:2])
    # Horizontal shimmer bands
    for y in range(0, img.height, random.randint(15, 30)):
        h = random.randint(1, 3)
        noise[y:y+h, :] += random.uniform(-6, 6)
    arr[:, :, :3] = np.clip(arr[:, :, :3] + noise[:, :, np.newaxis], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

#  INDIA WATERMARK (45°) 
def add_india_watermark(img):
    """
    Draws 'INDIA' diagonally at 45° across the plate in semi-transparent blue.
    This simulates the hot-stamp film on real HSRPs.
    Does NOT interfere with plate text recognition — it's behind the text layer.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(MAIN_FONT_PATH, 11)

    # Tile 'INDIA' text diagonally
    spacing_x, spacing_y = 55, 18
    for xi in range(-PLATE_W, PLATE_W * 2, spacing_x):
        for yi in range(-PLATE_H, PLATE_H * 2, spacing_y):
            # Rotate each text chunk via a temp image
            txt_img = Image.new("RGBA", (60, 14), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(txt_img)
            tdraw.text((0, 0), "INDIA", font=font, fill=(0, 80, 180, 50))
            rotated = txt_img.rotate(45, expand=True)
            overlay.paste(rotated, (xi, yi), rotated)

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")

#  ASHOKA CHAKRA HOLOGRAM (top-left) 
def add_hologram(draw, x, y, size=22):
    """Simple blue circle to simulate Ashoka Chakra hologram position."""
    draw.ellipse([x, y, x+size, y+size], fill=(0, 100, 200, 180), outline=(0, 60, 150), width=1)
    # Draw spokes like chakra
    cx, cy = x + size//2, y + size//2
    r = size // 2 - 2
    for spoke in range(24):
        angle = math.radians(spoke * 15)
        x1 = cx + int(r * 0.3 * math.cos(angle))
        y1 = cy + int(r * 0.3 * math.sin(angle))
        x2 = cx + int(r * math.cos(angle))
        y2 = cy + int(r * math.sin(angle))
        draw.line([x1, y1, x2, y2], fill=(0, 40, 120, 200), width=1)

#  IND TEXT (between hologram and plate number) 
def add_ind_text(draw, font_small):
    """Draws 'IND' in blue — purely decorative, positioned left, small."""
    draw.text((30, 8), "IND", font=font_small, fill=(0, 80, 180))

#  AUGMENTATIONS
def augment(img):
    # Gaussian noise
    if random.random() < 0.6:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(1, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # Blur (motion or gaussian)
    if random.random() < 0.4:
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        else:
            img = img.filter(ImageFilter.BoxBlur(random.uniform(0.3, 1.0)))

    # Slight rotation
    if random.random() < 0.5:
        angle = random.uniform(-4, 4)
        img = img.rotate(angle, fillcolor=img.getpixel((0, 0)), expand=False)

    # Perspective warp (simulate camera angle)
    if random.random() < 0.4:
        w, h = img.size
        skew = random.randint(3, 10)
        coeffs = find_perspective_coeffs(
            [(0,0),(w,0),(w,h),(0,h)],
            [(random.randint(0,skew), random.randint(0,skew)),
             (w-random.randint(0,skew), random.randint(0,skew)),
             (w-random.randint(0,skew), h-random.randint(0,skew)),
             (random.randint(0,skew), h-random.randint(0,skew))]
        )
        img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Shadow overlay
    if random.random() < 0.3:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        x1 = random.randint(0, PLATE_W // 2)
        alpha = random.randint(30, 80)
        draw.rectangle([x1, 0, x1 + random.randint(20, 100), PLATE_H],
                        fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Dirt spots
    if random.random() < 0.25:
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 8)):
            x = random.randint(0, PLATE_W)
            y = random.randint(0, PLATE_H)
            r = random.randint(1, 4)
            gray = random.randint(60, 160)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(gray, gray, gray))

    # Brightness jitter
    if random.random() < 0.5:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr * random.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img

def find_perspective_coeffs(src, dst):
    matrix = []
    for p1, p2 in zip(src, dst):
        matrix += [
            [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]],
            [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]],
        ]
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(dst).reshape(8)
    res = np.linalg.solve(A, B)
    return np.array(res).flatten()

#  PLATE GENERATORS

def make_hsrp_plate():
    """Modern HSRP Indian plate — Charles Wright style, reflective, with hologram & watermark."""
    bg_color, text_color, ptype, green_border = plate_type_and_colors()
    plate_text = generate_plate_text()

    img = Image.new("RGB", (PLATE_W, PLATE_H), bg_color)
    img = add_reflective_texture(img)
    img = add_india_watermark(img)

    draw = ImageDraw.Draw(img)

    # Border
    border_color = (0, 130, 60) if green_border else COLORS["border"]
    draw.rectangle([2, 2, PLATE_W-3, PLATE_H-3], outline=border_color, width=3)

    # Hologram (top-left)
    add_hologram(draw, 6, 6, size=20)

    # IND text (small, blue, after hologram)
    font_small = load_font(MAIN_FONT_PATH, 10)
    add_ind_text(draw, font_small)

    # Main plate number — centered, large
    font_size = random.randint(52, 58)
    font = load_font(MAIN_FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (PLATE_W - tw) // 2 + 10   # slight right offset for hologram
    ty = (PLATE_H - th) // 2

    # Embossed effect (slight shadow)
    draw.text((tx+1, ty+1), plate_text, font=font, fill=(100, 100, 100))
    draw.text((tx, ty), plate_text, font=font, fill=text_color)

    img = augment(img)
    return img, plate_text, f"india_hsrp_{ptype}"


def make_old_indian_plate():
    """Pre-2019 Indian plate — varied fonts, no hologram, sometimes colored border."""
    bg_color, text_color, ptype, _ = plate_type_and_colors()
    plate_text = generate_plate_text()

    # Old plates: sometimes white on black
    if random.random() < 0.1:
        bg_color = (20, 20, 20)
        text_color = COLORS["white_text"]

    img = Image.new("RGB", (PLATE_W, PLATE_H), bg_color)
    draw = ImageDraw.Draw(img)

    # Simple border
    border_colors = [(0,0,0),(0,0,180),(180,0,0)]
    draw.rectangle([2,2,PLATE_W-3,PLATE_H-3],
                   outline=random.choice(border_colors), width=2)

    # Old plates: random non-standard fonts
    old_fonts = [MONO_FONT_PATH, SERIF_FONT_PATH, SANS_FONT_PATH,
                 FREE_SANS_PATH, ROBOTO_FONT_PATH]
    font_path = random.choice([f for f in old_fonts if os.path.exists(f)])
    font_size = random.randint(44, 56)
    font = load_font(font_path, font_size)

    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (PLATE_W - tw) // 2
    ty = (PLATE_H - th) // 2

    draw.text((tx, ty), plate_text, font=font, fill=text_color)

    img = augment(img)
    return img, plate_text, f"india_old_{ptype}"


#  FOREIGN PLATE GENERATORS

FOREIGN_STYLES = {
    "eu_generic": {
        "bg": (255, 255, 255), "text": (10, 10, 10),
        "left_strip": (0, 50, 180),   # EU blue strip
        "format": lambda: f"{random.choice(['AB','CD','EF','GH'])}{random.randint(100,999)}{random.choice(['XY','KL','MN'])}"
    },
    "us_generic": {
        "bg": (255, 255, 255), "text": (10, 10, 10),
        "left_strip": None,
        "format": lambda: f"{random.choice(['CA','NY','TX','FL'])}{random.randint(100,999)}{random.choice(['ABC','XYZ','MNP'])}"
    },
    "uk_style": {
        "bg": (255, 220, 0), "text": (10, 10, 10),
        "left_strip": None,
        "format": lambda: f"{random.choice(['AB','CD','EF'])}{random.randint(10,99)}{random.choice(['ABC','XYZ','KLM'])}"
    },
    "gulf_style": {
        "bg": (255, 255, 255), "text": (10, 10, 10),
        "left_strip": None,
        "format": lambda: f"{random.randint(10000,99999)}"
    },
}

def make_foreign_plate():
    style_name = random.choice(list(FOREIGN_STYLES.keys()))
    style = FOREIGN_STYLES[style_name]
    plate_text = style["format"]()

    img = Image.new("RGB", (PLATE_W, PLATE_H), style["bg"])
    draw = ImageDraw.Draw(img)

    # EU blue strip on left
    if style["left_strip"]:
        draw.rectangle([0, 0, 28, PLATE_H], fill=style["left_strip"])
        font_eu = load_font(SANS_FONT_PATH, 9)
        draw.text((4, PLATE_H//2 - 5), "EU", font=font_eu, fill=(255,255,255))

    draw.rectangle([2, 2, PLATE_W-3, PLATE_H-3], outline=(80,80,80), width=2)

    # Foreign plates: varied fonts
    foreign_fonts = [CONDENSED_PATH, SERIF_FONT_PATH, MONO_FONT_PATH,
                     FREE_SANS_PATH, ROBOTO_FONT_PATH]
    font_path = random.choice([f for f in foreign_fonts if os.path.exists(f)])
    font_size = random.randint(46, 58)
    font = load_font(font_path, font_size)

    x_offset = 35 if style["left_strip"] else 0
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x_offset + (PLATE_W - x_offset - tw) // 2
    ty = (PLATE_H - th) // 2

    draw.text((tx, ty), plate_text, font=font, fill=style["text"])

    img = augment(img)
    return img, plate_text, f"foreign_{style_name}"


# main

def main(num_images=NUM_IMAGES, out_dir=OUT_DIR):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    pad = len(str(num_images))

    for i in range(num_images):
        r = random.random()

        if r < INDIA_RATIO:
            # Indian plate
            r2 = random.random()
            if r2 < INDIA_HSRP_RATIO:
                img, text, ptype = make_hsrp_plate()
            else:
                img, text, ptype = make_old_indian_plate()
        else:
            img, text, ptype = make_foreign_plate()

        fname = f"plate_{str(i).zfill(pad)}.jpg"
        img.save(os.path.join(img_dir, fname), quality=92)
        records.append({"filename": fname, "plate_text": text, "plate_type": ptype})

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_images} generated...")

    csv_path = os.path.join(out_dir, "annotations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","plate_text","plate_type"])
        writer.writeheader()
        writer.writerows(records)

    # Stats
    types = {}
    for r in records:
        types[r["plate_type"]] = types.get(r["plate_type"], 0) + 1

    print(f"\nDone! {num_images} plates → {out_dir}/")
    print(f"CSV: {csv_path}")
    print("\nDistribution:")
    for k, v in sorted(types.items()):
        print(f"  {k:<30} {v:>5}  ({v/num_images*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=NUM_IMAGES)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    args = parser.parse_args()
    main(args.num, args.out)
