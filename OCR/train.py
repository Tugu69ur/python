import csv
import torch
import evaluate
from PIL import Image
from torch.utils.data import Dataset


from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------
# Load processor + model
# ---------------------------

from dataclasses import dataclass


@dataclass
class OCRDataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        return {"pixel_values": pixel_values, "labels": labels}


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Important model configs
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size


# ---------------------------
# Dataset Class
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

        # Load image
        image = Image.open(f"{self.img_dir}/{sample['images']}").convert("RGB")

        # If vertical Mongolian script -> rotate if needed
        # image = image.rotate(90, expand=True)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(
            0
        )

        # Tokenize label
        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------
# Load datasets
# ---------------------------

train_dataset = MongolianOCRDataset("data/csv/train.csv", "data/train", processor)

val_dataset = MongolianOCRDataset("data/csv/val.csv", "data/val", processor)

# ---------------------------
# Evaluation metric
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
# Training arguments
# ---------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-mongolian",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=20,
    logging_steps=200,
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
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
# Train model
# ---------------------------

trainer.train()

# Save model
trainer.save_model("./trocr-mongolian-final")
processor.save_pretrained("./trocr-mongolian-final")

print("Training complete!")
