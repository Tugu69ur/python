import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("../mongolian_script_model.h5")
labels = {}
with open("../labels.txt", "r", encoding="utf-8") as f:
    for line in f:
        v, k = line.strip().split(":")
        labels[int(v)] = k

IMG_SIZE = 64
def preprocess_letter(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load word
word_img = cv2.imread("namor.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(word_img, 128, 255, cv2.THRESH_BINARY_INV)

# Vertical projection
v_proj = np.sum(thresh, axis=0)
thresh_val = 10  # adjust if needed
splits = []
in_letter = False
for i, val in enumerate(v_proj):
    if val > thresh_val and not in_letter:
        start = i
        in_letter = True
    elif val <= thresh_val and in_letter:
        end = i
        in_letter = False
        splits.append((start, end))
if in_letter:
    splits.append((start, len(v_proj)))

# Predict letters
predicted_letters = []
for start, end in splits:
    letter_img = thresh[:, start:end]
    letter_img = preprocess_letter(letter_img)
    pred = model.predict(letter_img)
    class_idx = np.argmax(pred)
    predicted_letters.append(labels[class_idx])

print("🧠 Predicted word:", "".join(predicted_letters))
