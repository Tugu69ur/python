import cv2
import os
import numpy as np

IMG_PATH = "mongolian_cons.gif"
OUT_DIR = "Mongolian_Dataset_Final"

# The 9 letters in each of the 3 main horizontal blocks
LETTERS = [
    ["N", "Ng", "B", "P", "Kh", "Gh", "M", "L", "H_soft"],
    ["G", "S", "Sh", "T", "D", "Ch", "J", "Y", "R"],
    ["V", "F", "Ch_Ts", "G_hard", "K", "Ts", "Z", "H", "Lkh"],
]
POSITIONS = ["Initial", "Medial", "Final"]

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError("Image not found!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

# 1. SPLIT INTO 3 MAIN HORIZONTAL BLOCKS
h_split = img.shape[0] // 3
for b_idx in range(3):
    block_th = th[b_idx * h_split : (b_idx + 1) * h_split, 60:]  # Skip row labels
    block_color = img[b_idx * h_split : (b_idx + 1) * h_split, 60:]

    # 2. FIND COLUMN GAPS (Horizontal Scan)
    col_sums = np.sum(block_th, axis=0)
    col_mask = col_sums > 0
    col_indices = []
    start = None
    for i, val in enumerate(col_mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 10:
                col_indices.append((start, i))
            start = None

    # 3. PROCESS EACH COLUMN (VERTICAL SCAN)
    for c_idx, (x1, x2) in enumerate(col_indices):
        if c_idx >= 9:
            break

        letter_name = LETTERS[b_idx][c_idx]
        letter_dir = os.path.join(OUT_DIR, letter_name)
        os.makedirs(letter_dir, exist_ok=True)

        col_th = block_th[:, x1:x2]
        col_color = block_color[:, x1:x2]

        # Find Row Gaps inside the column
        row_sums = np.sum(col_th, axis=1)
        row_mask = row_sums > 0
        row_indices = []
        r_start = None
        for i, val in enumerate(row_mask):
            if val and r_start is None:
                r_start = i
            elif not val and r_start is not None:
                if i - r_start > 8:
                    row_indices.append((r_start, i))
                r_start = None

        # 4. SAVE INITIAL, MEDIAL, FINAL
        for r_idx, (y1, y2) in enumerate(row_indices):
            if r_idx >= 3:
                break  # Ignore labels at the bottom

            crop = col_color[y1:y2, :]
            # Final tight crop to remove side-padding
            crop_th = col_th[y1:y2, :]
            coords = cv2.findNonZero(crop_th)
            if coords is not None:
                tx, ty, tw, th_box = cv2.boundingRect(coords)
                final_crop = crop[ty : ty + th_box, tx : tx + tw]

                # Naming
                fn = f"{POSITIONS[r_idx]}.png"
                cv2.imwrite(os.path.join(letter_dir, fn), final_crop)

print("✅ DONE. Check the 'Mongolian_Dataset_Final' folder.")
