from PIL import Image
import torch

model.eval()
img = Image.open("test_img.jpg").convert("L")
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(img)
    preds = out.argmax(2).squeeze(0).cpu().numpy()
    text = decode(preds, train_dataset.idx_to_char)
    print("🧾 Predicted:", text)
