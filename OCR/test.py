import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
processor = TrOCRProcessor.from_pretrained("./trocr-mongolian-final")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-mongolian-final")

model.to(device)
model.eval()

# Load test image
image = Image.open("data/test/dict-0.png").convert("RGB")

pixel_values = processor(
    image,
    return_tensors="pt",
    do_resize=True,
    size=(384, 384),
    do_normalize=True,
).pixel_values.to(device)

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=128,
        num_beams=4,
    )

predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Prediction:")
print(predicted_text)
