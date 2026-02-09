#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch CRNN + CTC training + prediction script for Mongolian vertical-script lines
Paths expected:
 - data/csv/train.csv, data/csv/val.csv, data/csv/test.csv
 - data/train/..., data/val/..., data/test/...

Usage:
    python pytorch_mongolian_ocr.py --mode train
    python pytorch_mongolian_ocr.py --mode predict --image data/test/lyrics-40000.png
"""

import os
import argparse
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------- Config -------------------------
DEFAULT_HEIGHT = 64
BLANK_INDEX = 0

CSV_DIR = "data/csv"
TRAIN_CSV = os.path.join(CSV_DIR, "train.csv")
VAL_CSV = os.path.join(CSV_DIR, "val.csv")
TEST_CSV = os.path.join(CSV_DIR, "test.csv")
TRAIN_IMG_DIR = "data/train"
VAL_IMG_DIR = "data/val"
TEST_IMG_DIR = "data/test"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------- Utilities -------------------------
def build_vocab_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    chars = set()
    for t in texts:
        for ch in t:
            chars.add(ch)
    return sorted(list(chars))

def save_training_plot(train_losses, val_losses, out_path="training_progress.png"):
    plt.figure(figsize=(8,6))
    plt.plot(range(1,len(train_losses)+1), train_losses, label='train loss')
    if len(val_losses) > 0:
        plt.plot(range(1,len(val_losses)+1), val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.title('Training Progress')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

# ------------------------- Dataset -------------------------
class MongolianDataset(Dataset):
    def __init__(self, csv_file, img_dir, char_to_idx, height=DEFAULT_HEIGHT, augment=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.height = height
        self.augment = augment

        base = [T.Grayscale(num_output_channels=1)]
        if augment:
            base += [T.RandomAffine(degrees=2, translate=(0.02,0.02), shear=2)]
        base += [T.ToTensor()]
        self.transform = T.Compose(base)

    def __len__(self):
        return len(self.df)

    def _open_and_resize(self, path):
        img = Image.open(path).convert('L')
        w, h = img.size
        new_h = self.height
        new_w = max(1,int(w*(new_h/float(h))))
        img = img.resize((new_w,new_h), Image.BILINEAR)
        return img

    def text_to_labels(self, text):
        return [self.char_to_idx.get(ch,0) for ch in text]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['images']).strip())
        img = self._open_and_resize(img_path)
        img = self.transform(img)
        label = torch.tensor(self.text_to_labels(str(row['text']).strip()), dtype=torch.long)
        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)
    batch_size = len(imgs)
    H = imgs[0].shape[1]
    max_w = max([img.shape[2] for img in imgs])

    padded = torch.zeros((batch_size,1,H,max_w), dtype=imgs[0].dtype)
    for i,img in enumerate(imgs):
        w = img.shape[2]
        padded[i,:, :, :w] = img

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels) if len(labels) > 0 else torch.tensor([], dtype=torch.long)
    return padded, labels_concat, label_lengths

# ------------------------- CRNN Model -------------------------
class CRNN(nn.Module):
    def __init__(self, n_classes:int, imgH:int=DEFAULT_HEIGHT, nc:int=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )
        # Compute reduced height
        reduced_h = max(1, (imgH // 16) - 1)
        rnn_input_size = 512 * reduced_h
        self.rnn = nn.LSTM(rnn_input_size,256,num_layers=2,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self,x):
        x = self.cnn(x)
        b,c,h,w = x.size()
        assert h>=1, f"Feature map height <1: {h}"
        x = x.permute(0,3,1,2).contiguous().view(b,w,c*h)
        x,_ = self.rnn(x)
        x = self.fc(x)
        x = x.log_softmax(2)
        return x

# ------------------------- Decoding -------------------------
def ctc_greedy_decode(preds:np.ndarray, idx_to_char:dict):
    results = []
    for b in range(preds.shape[0]):
        seq = np.argmax(preds[b],axis=1).tolist()
        out=[]
        prev=-1
        for p in seq:
            if p != prev and p!=BLANK_INDEX:
                out.append(idx_to_char.get(p,''))
            prev=p
        results.append(''.join(out))
    return results

# ------------------------- Training -------------------------
def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss=0
    n_batches=0
    pbar = tqdm(dataloader, desc='Train', leave=False)
    for imgs, labels_concat, label_lengths in pbar:
        imgs = imgs.to(device)
        labels_concat = labels_concat.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(imgs).permute(1,0,2)
        input_lengths = torch.full((outputs.size(1),),outputs.size(0),dtype=torch.long).to(device)
        loss = criterion(outputs, labels_concat, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        optimizer.step()

        running_loss+=loss.item()
        n_batches+=1
        pbar.set_postfix(loss=running_loss/max(1,n_batches))
    return running_loss/max(1,n_batches)

def validate(model,dataloader,criterion,device):
    model.eval()
    running_loss=0
    n_batches=0
    with torch.no_grad():
        pbar=tqdm(dataloader,desc='Val',leave=False)
        for imgs, labels_concat, label_lengths in pbar:
            imgs = imgs.to(device)
            labels_concat = labels_concat.to(device)
            label_lengths = label_lengths.to(device)
            outputs = model(imgs).permute(1,0,2)
            input_lengths = torch.full((outputs.size(1),),outputs.size(0),dtype=torch.long).to(device)
            loss = criterion(outputs, labels_concat, input_lengths, label_lengths)
            running_loss+=loss.item()
            n_batches+=1
            pbar.set_postfix(val_loss=running_loss/max(1,n_batches))
    return running_loss/max(1,n_batches)

def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:',device)

    chars = build_vocab_from_csv(args.train_csv)
    print(f'Found {len(chars)} unique characters')
    char_to_idx = {ch:i+1 for i,ch in enumerate(chars)}
    idx_to_char = {i+1:ch for i,ch in enumerate(chars)}
    idx_to_char[0]=''

    train_ds = MongolianDataset(args.train_csv,args.train_dir,char_to_idx,height=args.height,augment=True)
    val_ds = MongolianDataset(args.val_csv,args.val_dir,char_to_idx,height=args.height,augment=False)
    train_loader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=4)
    val_loader = DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn,num_workers=4)

    model = CRNN(n_classes=len(chars)+1,imgH=args.height).to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX,zero_infinity=True)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    train_losses=[]
    val_losses=[]
    best_val=float('inf')

    for epoch in range(1,args.epochs+1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        train_loss=train_one_epoch(model,train_loader,criterion,optimizer,device)
        val_loss=validate(model,val_loader,criterion,device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        ckpt_path=os.path.join(CHECKPOINT_DIR,f'crnn_epoch{epoch}.pth')
        torch.save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'char_to_idx':char_to_idx,'idx_to_char':idx_to_char},ckpt_path)
        if val_loss<best_val:
            best_val=val_loss
            best_path=os.path.join(CHECKPOINT_DIR,'best_crnn.pth')
            torch.save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'char_to_idx':char_to_idx,'idx_to_char':idx_to_char},best_path)
        save_training_plot(train_losses,val_losses,'training_progress.png')

# ------------------------- Prediction -------------------------
def predict_image(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path=args.checkpoint
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path,map_location=device)
    idx_to_char=ckpt['idx_to_char']
    char_to_idx=ckpt['char_to_idx']
    n_classes=len(idx_to_char)
    model=CRNN(n_classes=n_classes,imgH=args.height).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    img=Image.open(args.image).convert('L')
    w,h=img.size
    new_h=args.height
    new_w=max(1,int(w*(new_h/float(h))))
    img=img.resize((new_w,new_h),Image.BILINEAR)
    transform=T.Compose([T.Grayscale(num_output_channels=1),T.ToTensor()])
    tensor=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds=model(tensor)
        preds_np=preds.cpu().numpy()
        texts=ctc_greedy_decode(preds_np,idx_to_char)
    print('Prediction:',texts[0])
    return texts[0]

# ------------------------- CLI -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['train','predict'],required=True)
    parser.add_argument('--train_csv',default=TRAIN_CSV)
    parser.add_argument('--val_csv',default=VAL_CSV)
    parser.add_argument('--test_csv',default=TEST_CSV)
    parser.add_argument('--train_dir',default=TRAIN_IMG_DIR)
    parser.add_argument('--val_dir',default=VAL_IMG_DIR)
    parser.add_argument('--test_dir',default=TEST_IMG_DIR)
    parser.add_argument('--height',type=int,default=DEFAULT_HEIGHT)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--image',type=str,default=None)
    parser.add_argument('--checkpoint',type=str,default=os.path.join(CHECKPOINT_DIR,'best_crnn.pth'))

    args = parser.parse_args()
    if args.mode=='train':
        train_loop(args)    
    elif args.mode=='predict':
        assert args.image is not None, "--image is required in predict mode"
        predict_image(args)

if __name__=='__main__':
    main()
