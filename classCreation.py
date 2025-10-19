import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MongolianDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.alphabet =" ᠨ ᠡ ᠮ ᠡ ᠭ ᠦ ᠦ   ᠵ ᠣ ᠪ ᠠ ᠷ ᠢ   ᠻ ᠣ ᠨ ᠲ ᠷ ᠠ ᠪ ᠠ ᠨ ᠳ ᠠ   ᠳ ᠠ ᠯ ᠠ ᠩ   ᠴ ᠦ ᠷ ᠳ ᠡ ᠯ ᠵ ᠡ ᠭ ᠦ ᠯ ᠬ ᠦ   ᠠ ᠭ ᠎ ᠠ   ᠳ ᠣ ᠷ ᠣ ᠢ   ᠠ ᠯ ᠠ ᠭ"
        self.char_to_idx = {c:i+1 for i,c in enumerate(self.alphabet)}  # 0=blank
        self.idx_to_char = {i:c for c,i in self.char_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['images']}"
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        text = row['text']
        label = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
