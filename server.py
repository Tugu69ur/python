# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import numpy as np
from train import CRNN, decode  # your model code

# ------------------------- Settings -------------------------
alphabet = [
    " ", "ᠠ","ᠡ","ᠢ","ᠣ","ᠤ","ᠥ","ᠦ","ᠨ","ᠩ","ᠪ","ᠫ","ᠬ","ᠭ",
    "ᠮ","ᠯ","ᠰ","ᠱ","ᠲ","ᠳ","ᠴ","ᠵ","ᠶ","ᠷ","ᠸ","ᠹ","ᠺ",
    "ᠻ","ᠼ","ᠽ","ᠾ","ᠿ"
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(nClasses=len(alphabet)+1).to(device)
checkpoint = torch.load("checkpoint_epoch13.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------------- API -------------------------
app = FastAPI()

# CORS so React Native app can call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        preds = output.argmax(2).squeeze(0).cpu().numpy()
        text = decode(preds)

    return {"result": text}

# Run: uvicorn server:app --reload --host 0.0.0.0 --port 8000
