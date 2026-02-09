import cv2
import numpy as np
from PIL import Image
import os

ROOT_DIR = "A"
OUTPUT_ROOT = "A_fixed_cropped"
PADDING = 1  # pixels
MAX_ANGLE = 12  # degrees

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def deskew_and_crop_safe(path):
    # ---- READ ----
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    h, w = gray.shape

    # ---- BINARIZE ----
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(bw > 0))

    # ---- DESKEW (SAFE) ----
    angle = 0
    if len(coords) > 40 and h >= 40 and w >= 40:
        raw_angle = cv2.minAreaRect(coords)[-1]
        if raw_angle < -45:
            raw_angle = -(90 + raw_angle)
        else:
            raw_angle = -raw_angle

        if abs(raw_angle) <= MAX_ANGLE:
            angle = raw_angle

    pil = Image.open(path).convert("RGBA")
    rotated = pil.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    # ---- CROP TO INK ----
    arr = np.array(rotated)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rotated

    x1 = max(xs.min() - PADDING, 0)
    y1 = max(ys.min() - PADDING, 0)
    x2 = min(xs.max() + PADDING, rotated.width)
    y2 = min(ys.max() + PADDING, rotated.height)

    return rotated.crop((x1, y1, x2, y2))


# ---- PROCESS ALL FOLDERS ----
for folder in os.listdir(ROOT_DIR):
    src_dir = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(src_dir):
        continue

    dst_dir = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(dst_dir, exist_ok=True)

    for file in os.listdir(src_dir):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)

            result = deskew_and_crop_safe(src)
            if result:
                result.save(dst)
                print(f"✅ {folder}/{file}")
            else:
                print(f"⚠️ Skipped {folder}/{file}")

print("🎉 All letters fixed, deskewed, and ink-cropped")
