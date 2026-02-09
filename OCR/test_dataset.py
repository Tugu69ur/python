from transformers import TrOCRProcessor
from PIL import Image
import csv

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

csv_path = "data/csv/train.csv"
img_dir = "data/train"

with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    row = next(reader)

image = Image.open(f"{img_dir}/{row['images']}").convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values

print("Loaded image shape:", pixel_values.shape)
print("Text sample:", row["text"])
