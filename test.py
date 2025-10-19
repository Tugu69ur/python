from train import CRNN
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np

# -------------------------
# Alphabet
# -------------------------
alphabet = [
    " ", "ᠠ","ᠡ","ᠢ","ᠣ","ᠤ","ᠥ","ᠦ","ᠨ","ᠩ","ᠪ","ᠫ","ᠬ","ᠭ",
    "ᠮ","ᠯ","ᠰ","ᠱ","ᠲ","ᠳ","ᠴ","ᠵ","ᠶ","ᠷ","ᠸ","ᠹ","ᠺ",
    "ᠻ","ᠼ","ᠽ","ᠾ","ᠿ"
]

# -------------------------
# Decode helper
# -------------------------
def decode(preds):
    result = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:
            result.append(alphabet[p-1])
        prev = p
    return "".join(result)

# -------------------------
# Image transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------
# Device & Model
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(nClasses=len(alphabet)+1).to(device)

checkpoint = torch.load("checkpoint_epoch13.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# -------------------------
# Find most similar image
# -------------------------
def find_most_similar_image(uploaded_image_path, csv_file, img_dir):
    df = pd.read_csv(csv_file)
    uploaded = Image.open(uploaded_image_path).convert("L").resize((128, 32))
    uploaded_np = np.array(uploaded, dtype=np.float32)

    best_match = None
    best_score = float("inf")

    for _, row in df.iterrows():
        train_path = os.path.join(img_dir, row["images"])
        if not os.path.exists(train_path):
            continue
        try:
            img = Image.open(train_path).convert("L").resize((128, 32))
            img_np = np.array(img, dtype=np.float32)
            diff = np.mean(np.abs(uploaded_np - img_np))
            if diff < best_score:
                best_score = diff
                best_match = row
        except:
            continue

    if best_match is not None and best_score < 10:  # threshold, adjust if needed
        print(f"✅ Found similar image: {best_match['images']} (diff={best_score:.2f})")
        print(f"📝 Text: {best_match['text']}")
        return True
    return False

# -------------------------
# Predict function
# -------------------------
def predict(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        preds = output.argmax(2).squeeze(0).cpu().numpy()
        return decode(preds)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    img_path = "data/train/lyrics-39999.png"
    csv_path = "data/csv/train.csv"
    img_dir = "data/train"

    # Step 1: Try to find a visually similar image
    if not find_most_similar_image(img_path, csv_path, img_dir):
        # Step 2: If no similar image found, use CRNN prediction
        print("🤖 Using model prediction instead...")
        print("Prediction:", predict(img_path))
