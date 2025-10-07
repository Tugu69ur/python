from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("mongolian_script_model.h5")

# Load labels
labels = {}
with open("labels.txt", "r", encoding="utf-8") as f:
    for line in f:
        idx, name = line.strip().split(": ")
        labels[int(idx)] = name

# Load and process image
img = Image.open("dataset/test/0000.png").convert("RGB")  # <-- convert to RGB
img = img.resize((64, 64))
img = img_to_array(img)
img = np.expand_dims(img, axis=0) / 255.0

# Predict
pred = model.predict(img)
pred_class = np.argmax(pred[0])
print(f"Predicted class: {labels[pred_class]} (confidence: {pred[0][pred_class]:.2f})")
