import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm

# -------------------------
# Dataset
# -------------------------
class MongolianDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.alphabet = [
            " ", "ᠠ","ᠡ","ᠢ","ᠣ","ᠤ","ᠥ","ᠦ","ᠨ","ᠩ","ᠪ","ᠫ","ᠬ","ᠭ",
            "ᠮ","ᠯ","ᠰ","ᠱ","ᠲ","ᠳ","ᠴ","ᠵ","ᠶ","ᠷ","ᠸ","ᠹ","ᠺ",
            "ᠻ","ᠼ","ᠽ","ᠾ","ᠿ"
        ]
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.alphabet)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['images'])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        text = row['text'].replace(" ", " ").replace("\u202F", " ").strip()
        label = [self.char_to_idx.get(c, 0) for c in text]
        return image, torch.tensor(label, dtype=torch.long)

# -------------------------
# Collate Function
# -------------------------
def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return imgs, labels, label_lengths

# -------------------------
# CRNN Model
# -------------------------
class CRNN(nn.Module):
    def __init__(self, imgH=32, nChannels=1, nClasses=34):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nChannels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, nClasses)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1, f"Expected height=1 after CNN, got {h}"
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# -------------------------
# Decode helper
# -------------------------
def decode(preds, idx_to_char):
    result = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:
            result.append(idx_to_char.get(p, ""))
        prev = p
    return "".join(result)

# -------------------------
# Training
# -------------------------
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    csv_path = "data/csv/train.csv"
    img_dir = "data/train"

    train_dataset = MongolianDataset(csv_path, img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    model = CRNN(nClasses=len(train_dataset.alphabet) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    start_epoch = 7  # continue from epoch 7
    checkpoint_path = f"checkpoint_epoch{start_epoch}.pth"

    # -------------------------
    # Load checkpoint (if exists)
    # -------------------------
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            print(f"✅ Loaded checkpoint from {checkpoint_path} (epoch {start_epoch})")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights only from {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found, starting from scratch.")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for imgs, labels, label_lengths in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((imgs.size(0),), outputs.size(0), dtype=torch.long)
            loss = criterion(outputs, labels, input_lengths, label_lengths)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # -------------------------
        # Preview a sample prediction
        # -------------------------
        model.eval()
        with torch.no_grad():
            sample_img, _ = train_dataset[0]
            out = model(sample_img.unsqueeze(0).to(device))
            preds = out.argmax(2).squeeze(0).cpu().numpy()
            print("🧾 Sample:", decode(preds, train_dataset.idx_to_char))

        # -------------------------
        # Save checkpoint
        # -------------------------
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1
        }, f"checkpoint_epoch{epoch+1}.pth")
        print(f"💾 Saved: checkpoint_epoch{epoch+1}.pth")

    print("🎉 Training complete!")
