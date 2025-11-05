#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ABSA Model
Train model mới trên training data và lưu checkpoint
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# === Cấu hình ===
TRAINING_DATA_DIR = "/opt/airflow/projects/absa_streaming/training_data"
MODEL_NAME = "xlm-roberta-base"
TOKENIZER_LOCAL_DIR = "/opt/airflow/models/hf-cache/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
OLD_MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
NEW_MODEL_PATH = "/opt/airflow/models/new_absa_model.pt"
MAX_LEN = 64
EPOCHS = 1  # Demo: chỉ 1 epochs
BATCH_SIZE = 4
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASPECTS = ["price", "shipping", "outlook", "quality", "size", "shop_service", "general", "others"]

# === Model Definition ===
class ABSAModel(nn.Module):
    def __init__(self, backbone_path, num_aspects=8):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_m = nn.Linear(H, num_aspects)
        self.head_s = nn.Linear(H, num_aspects * 3)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

# === Dataset ===
class ABSADataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=64):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        labels = [sample[asp] for asp in ASPECTS]  # 0=NEG, 1=NEU, 2=POS
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    print("=" * 80, flush=True)
    print("[TRAIN] Starting model training...", flush=True)
    print("=" * 80, flush=True)
    
    # Load data
    train_path = os.path.join(TRAINING_DATA_DIR, "train.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    
    print(f"[TRAIN] Loaded {len(train_samples)} training samples", flush=True)
    
    # Load tokenizer
    print(f"[TRAIN] Loading tokenizer from {TOKENIZER_LOCAL_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_DIR, use_fast=True, local_files_only=True)
    
    # Create dataset
    train_dataset = ABSADataset(train_samples, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load old model as starting point
    print(f"[TRAIN] Loading base model from {OLD_MODEL_PATH}", flush=True)
    model = ABSAModel(TOKENIZER_LOCAL_DIR, num_aspects=len(ASPECTS))
    old_state = torch.load(OLD_MODEL_PATH, map_location=DEVICE)
    if isinstance(old_state, dict) and "state_dict" in old_state:
        old_state = old_state["state_dict"]
    model.load_state_dict(old_state, strict=False)
    model.to(DEVICE)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    total_loss = 0

    STEP_LOG_INTERVAL = 1 # In log sau mỗi N steps
    
    for epoch in range(EPOCHS):
        print(f"\n[TRAIN] Epoch {epoch+1}/{EPOCHS}", flush=True)
        epoch_loss = 0
        
        # Thêm enumerate để lấy index (step_count)
        for step_count, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            _, logits_s = model(input_ids, attention_mask)
            
            # Loss: cross-entropy trên sentiment predictions
            loss = 0
            for i in range(len(ASPECTS)):
                loss += criterion(logits_s[:, i, :], labels[:, i])
            loss = loss / len(ASPECTS)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # BỔ SUNG: IN LOG SAU MỖI N STEPS
            if (step_count + 1) % STEP_LOG_INTERVAL == 0:
                print(f"[TRAIN] --> Step {step_count+1} / {len(train_loader)} | Current Batch Loss: {loss.item():.4f}", flush=True)
            
        avg_loss = epoch_loss / len(train_loader)
        total_loss += avg_loss
        print(f"[TRAIN] Epoch {epoch+1} - FINAL Loss: {avg_loss:.4f}", flush=True)
    
    # Save new model
    print(f"\n[TRAIN] Saving new model to {NEW_MODEL_PATH}", flush=True)
    torch.save(model.state_dict(), NEW_MODEL_PATH)
    
    # Save training metadata
    metadata = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "train_samples": len(train_samples),
        "avg_loss": total_loss / EPOCHS,
    }
    metadata_path = NEW_MODEL_PATH.replace(".pt", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[TRAIN] ✅ Training completed!", flush=True)
    print(f"[TRAIN] ✅ Model saved: {NEW_MODEL_PATH}", flush=True)
    print(f"[TRAIN] ✅ Metadata saved: {metadata_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()


