import csv
import torch
import evaluate
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------
# Load base model
# ---------------------------

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# IMPORTANT CONFIG
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


# ---------------------------
# Dataset
# ---------------------------


class MongolianOCRDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, max_length=128):
        self.samples = []
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(f"{self.img_dir}/{sample['images']}").convert("RGB")

        pixel_values = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            size=(384, 384),
            do_normalize=True,
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


@dataclass
class OCRDataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------
# Load Data
# ---------------------------

train_dataset = MongolianOCRDataset(
    "data/csv/train.csv",
    "data/train",
    processor,
)

val_dataset = MongolianOCRDataset(
    "data/csv/val.csv",
    "data/val",
    processor,
)

# ---------------------------
# Metric
# ---------------------------

cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# ---------------------------
# Training Arguments
# ---------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-mongolian",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    logging_steps=200,
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=10,
    learning_rate=1e-4,
    warmup_steps=2000,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# ---------------------------
# Trainer
# ---------------------------

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=OCRDataCollator(),
    compute_metrics=compute_metrics,
)

# ---------------------------
# Train
# ---------------------------

trainer.train()

# Save final model
trainer.save_model("./trocr-mongolian-final")
processor.save_pretrained("./trocr-mongolian-final")

print("Training complete.")
