#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Training Data for Model Retraining
Lấy dữ liệu từ PostgreSQL (streaming results) hoặc CSV để chuẩn bị training
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Training Data for Model Retraining
Lấy dữ liệu từ CSV (train và val độc lập) để chuẩn bị training
"""
import pandas as pd
import json
import os
import random

# === Cấu hình ===
CSV_PATH = "/opt/airflow/projects/absa_streaming/data/train_data.csv"
VAL_CSV_PATH = "/opt/airflow/projects/absa_streaming/data/val_data.csv" 
OUTPUT_DIR = "/opt/airflow/projects/absa_streaming/training_data"
SAMPLE_RATE = 0.1 
RANDOM_SEED = 42 # Seed cho việc lấy mẫu ngẫu nhiên nhất quán

ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
# Mapping labels: -1→NEG(0), 0→NEU(1), 1→POS(2), 2→POS(2)
LABEL_MAP = {-1: 0, 0: 1, 1: 2, 2: 2}

# === Hàm xử lý chung (Lọc, Parse, Sampling) ===
def process_data(csv_path, is_training_data, sample_rate, random_seed):
    
    if not os.path.exists(csv_path):
        print(f"[PREPARE DATA] ERROR: CSV file not found at {csv_path}", flush=True)
        return []

    df = pd.read_csv(csv_path)
    print(f"[PREPARE DATA] Loaded {len(df)} samples from {csv_path}")
    
    valid_samples = []
    skipped_empty_text = 0
    
    for idx, row in df.iterrows():
        text = row["Review"]
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            skipped_empty_text += 1
            continue
        
        sample = {"text": text.strip()}
        for asp in ASPECTS:
            val = int(row[asp])
            if val in LABEL_MAP:
                sample[asp.lower()] = LABEL_MAP[val]
            else:
                sample[asp.lower()] = 1
        
        valid_samples.append(sample)
    
    total_valid_samples = len(valid_samples)
    print(f"[PREPARE DATA] Found {total_valid_samples} valid samples in {os.path.basename(csv_path)}", flush=True)
    
    # Áp dụng Sampling
    if is_training_data and sample_rate < 1.0 and total_valid_samples > 0:
        random.seed(random_seed)
        sample_size = int(total_valid_samples * sample_rate)
        valid_samples = random.sample(valid_samples, sample_size)
        print(f"[PREPARE DATA] Sampled down to {len(valid_samples)} samples ({sample_rate*100:.0f}%)", flush=True)
    
    return valid_samples

def main():
    print("=" * 80)
    print("[PREPARE DATA] Starting...")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Xử lý Tập Train (từ train_data.csv, có sampling)
    train_samples = process_data(CSV_PATH, is_training_data=True, sample_rate=SAMPLE_RATE, random_seed=None)
    
    # 2. Xử lý Tập Validation (từ val_data.csv, không sampling)
    # val_samples = process_data(VAL_CSV_PATH, is_training_data=False, sample_rate=1.0, random_seed=None)
    val_samples = process_data(VAL_CSV_PATH, is_training_data=False, sample_rate=1.0, random_seed=None)
    # Lưu ra file
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    val_path = os.path.join(OUTPUT_DIR, "val.json")
    
    # Lưu train.json
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"[PREPARE DATA] ✅ Train: {len(train_samples)} samples → {train_path}")

    # Lưu val.json
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"[PREPARE DATA] ✅ Val: {len(val_samples)} samples → {val_path}")
    
    # Lưu metadata
    metadata = {
        "total_samples": len(train_samples) + len(val_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "aspects": [asp.lower() for asp in ASPECTS],
    }
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[PREPARE DATA] ✅ Metadata → {metadata_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()