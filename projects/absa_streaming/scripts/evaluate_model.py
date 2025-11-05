#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate ABSA Model
Đánh giá model trên validation set và lưu metrics
"""
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
import os
import sys

# === Cấu hình ===
TRAINING_DATA_DIR = "/opt/airflow/projects/absa_streaming/training_data"
MODEL_NAME = "xlm-roberta-base"
TOKENIZER_LOCAL_DIR = "/opt/airflow/models/hf-cache/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
MAX_LEN = 64
BATCH_SIZE = 16
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
        labels = [sample[asp] for asp in ASPECTS]
        
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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            _, logits_s = model(input_ids, attention_mask)
            preds = torch.argmax(logits_s, dim=-1)  # [batch, aspects]
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    all_preds = np.concatenate(all_preds, axis=0)  # [N, aspects]
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics per aspect
    metrics = {}
    for i, asp in enumerate(ASPECTS):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        f1 = f1_score(all_labels[:, i], all_preds[:, i], average="macro", zero_division=0)
        metrics[asp] = {"accuracy": float(acc), "f1": float(f1)}
    
    # Overall metrics
    overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    overall_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average="macro", zero_division=0)
    metrics["overall"] = {"accuracy": float(overall_acc), "f1": float(overall_f1)}
    
    return metrics

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/airflow/models/best_absa_hardshare.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else model_path.replace(".pt", "_eval.json")
    
    print("=" * 80, flush=True)
    print(f"[EVAL] Evaluating model: {model_path}", flush=True)
    print("=" * 80, flush=True)
    
    # Thêm LOGIC KIỂM TRA FILE ĐÃ TỒN TẠI, thoát khỏi main (ft01)
    if os.path.exists(output_path):
        print("=" * 80, flush=True)
        print(f"[EVAL] Metrics file already exists at {output_path}", flush=True)
        print("[EVAL] Skipping evaluation to save time.", flush=True)
        print("=" * 80, flush=True)
        return 

    # Load validation data
    val_path = os.path.join(TRAINING_DATA_DIR, "val.json")
    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    
    print(f"[EVAL] Loaded {len(val_samples)} validation samples", flush=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_DIR, use_fast=True, local_files_only=True)
    
    # Create dataset
    val_dataset = ABSADataset(val_samples, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    print(f"[EVAL] Loading model from {model_path}", flush=True)
    model = ABSAModel(TOKENIZER_LOCAL_DIR, num_aspects=len(ASPECTS))
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    
    # Evaluate
    print("[EVAL] Running evaluation...", flush=True)
    metrics = evaluate(model, val_loader, DEVICE)
    
    # Print results
    print("\n[EVAL] Results:", flush=True)
    for asp, m in metrics.items():
        if asp != "overall":
            print(f"  {asp:15s} → Acc: {m['accuracy']:.4f}, F1: {m['f1']:.4f}", flush=True)
    print("-" * 80, flush=True)
    print(f"  {'OVERALL':15s} → Acc: {metrics['overall']['accuracy']:.4f}, F1: {metrics['overall']['f1']:.4f}", flush=True)
    
    # Save metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[EVAL] ✅ Metrics saved to {output_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()


