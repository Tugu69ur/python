from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from random import choice
import torch
import torchvision.transforms as transforms
import io
from train import CRNN, decode  # CRNN class + decode function

# ------------------------- Alphabet & Map -------------------------
alphabet = [
    " ", "ᠠ","ᠡ","ᠢ","ᠣ","ᠤ","ᠥ","ᠦ","ᠨ","ᠩ","ᠪ","ᠫ","ᠬ","ᠭ",
    "ᠮ","ᠯ","ᠰ","ᠱ","ᠲ","ᠳ","ᠴ","ᠵ","ᠶ","ᠷ","ᠸ","ᠹ","ᠺ",
    "ᠻ","ᠼ","ᠽ","ᠾ","ᠿ"
]
idx_to_char = {i+1: c for i, c in enumerate(alphabet)}

# ------------------------- Device & Model -------------------------
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

# ------------------------- FastAPI -------------------------
app = FastAPI()

# CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            preds = output.argmax(2).squeeze(0).cpu().numpy()
            text = decode(preds, idx_to_char)

        return {"result": text}

    except Exception as e:
        import traceback
        print("OCR Server Error:", e)
        traceback.print_exc()
        return {"result": "Internal Server Error"}

# # BLACKJACKKKK
# SUITS = ["♠", "♥", "♦", "♣"]
# RANKS = [
#     {"r": "A", "v": 1},
#     {"r": "2", "v": 2},
#     {"r": "3", "v": 3},
#     {"r": "4", "v": 4},
#     {"r": "5", "v": 5},
#     {"r": "6", "v": 6},
#     {"r": "7", "v": 7},
#     {"r": "8", "v": 8},
#     {"r": "9", "v": 9},
#     {"r": "10", "v": 10},
#     {"r": "J", "v": 10},
#     {"r": "Q", "v": 10},
#     {"r": "K", "v": 10},
# ]

# def make_deck():
#     return [{"rank": r["r"], "suit": s, "value": r["v"]} for s in SUITS for r in RANKS]

# @app.get("/blackjack-hand")
# def blackjack_hand():
#     deck = make_deck()
#     hand = [deck.pop(choice(range(len(deck)))) for _ in range(2)]
#     return {"hand": hand}

# Run: uvicorn server:app --reload --host 0.0.0.0 --port 8000
