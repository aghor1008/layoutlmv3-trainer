import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Tokenizer,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

# Custom Dataset Class
class CustomMultiDocDataset(Dataset):
    def __init__(self, json_file, base_dir, tokenizer):
        self.base_dir = base_dir  # Folder containing JSON and images
        self.tokenizer = tokenizer

        # Load annotations from JSON
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = [img['id'] for img in self.annotations['images']]
        self.img_map = {img['id']: img for img in self.annotations['images']}
        self.category_map = {cat['id']: cat['name'] for cat in self.annotations['categories']}

        self.annotation_map = {}
        for ann in self.annotations['annotations']:
            if ann['image_id'] not in self.annotation_map:
                self.annotation_map[ann['image_id']] = []
            self.annotation_map[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.img_map[image_id]

        image_path = os.path.join(self.base_dir, img_info['file_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = img_info['width'], img_info['height']

        # Prepare annotations
        annotations = self.annotation_map.get(image_id, [])
        words = [self.category_map[ann['category_id']] for ann in annotations]
        boxes = [self._normalize_bbox(ann['bbox'], width, height) for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        encoding = self.tokenizer(
            text=words,
            boxes=boxes,
            images=image,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "bbox": encoding["bbox"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _normalize_bbox(self, bbox, width, height):
        x, y, w, h = bbox
        return [
            int((x / width) * 1000),
            int((y / height) * 1000),
            int(((x + w) / width) * 1000),
            int(((y + h) / height) * 1000)
        ]

# Metrics Function with Detailed Analysis
def compute_metrics(p):
    pred_labels = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids
    pred_labels = pred_labels.flatten()
    true_labels = true_labels.flatten()

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report for class-wise analysis
    report = classification_report(true_labels, pred_labels, zero_division=1)
    print("Classification Report:\n", report)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Load tokenizer and model
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=9)
model.to(device)

# Initialize datasets
base_dir = r"E:\layoutlmv3_project\data\pan1"
train_json = os.path.join(base_dir, "train", "annotations.coco.json")
valid_json = os.path.join(base_dir, "valid", "annotations.coco.json")
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

train_dataset = CustomMultiDocDataset(json_file=train_json, base_dir=train_dir, tokenizer=tokenizer)
valid_dataset = CustomMultiDocDataset(json_file=valid_json, base_dir=valid_dir, tokenizer=tokenizer)

# Training arguments with Learning Rate Scheduler
training_args = TrainingArguments(
    output_dir=r"E:\layoutlmv3_project\training output\results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Set save strategy to "epoch"
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=r"E:\layoutlmv3_project\training output\logs",
    logging_steps=100,
    fp16=True,
    load_best_model_at_end=True,
    save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer with advanced metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training with loss curve tracking
train_loss_values = []
eval_loss_values = []

for epoch in range(int(training_args.num_train_epochs)):
    trainer.train()
    train_metrics = trainer.evaluate(train_dataset)
    eval_metrics = trainer.evaluate(valid_dataset)

    train_loss_values.append(train_metrics["eval_loss"])
    eval_loss_values.append(eval_metrics["eval_loss"])

    # Plot Loss Curve
    plt.plot(range(1, epoch+2), train_loss_values, label="Training Loss")
    plt.plot(range(1, epoch+2), eval_loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

# Save the model and tokenizer
model.save_pretrained("E:/layoutlmv3_project/trainedmodel_new")
tokenizer.save_pretrained("E:/layoutlmv3_project/trainedmodel_new")
