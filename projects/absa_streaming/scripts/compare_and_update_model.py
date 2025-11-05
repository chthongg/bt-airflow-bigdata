#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Models and Update if Better
So sánh model mới với model cũ, chỉ cập nhật nếu tốt hơn. Ghi lịch sử cập nhật vào file JSON Lines.
"""
import json
import shutil
import os
from datetime import datetime

# === Cấu hình ===
OLD_MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
NEW_MODEL_PATH = "/opt/airflow/models/new_absa_model.pt"
BACKUP_DIR = "/opt/airflow/models/backups"
OLD_EVAL_PATH = OLD_MODEL_PATH.replace(".pt", "_eval.json")
NEW_EVAL_PATH = NEW_MODEL_PATH.replace(".pt", "_eval.json")
# 🌟 TÊN FILE LỊCH SỬ MỚI 🌟
HISTORY_LOG_PATH = os.path.join(BACKUP_DIR, "update_history.jsonl") 


def count_backup_files(backup_dir):
    """Đếm tổng số file .pt trong thư mục backup."""
    if not os.path.exists(backup_dir):
        return 0
    # Lọc và đếm các file kết thúc bằng .pt
    count = sum(1 for item in os.listdir(backup_dir) if item.endswith('.pt'))
    return count

def main():
    print("=" * 80, flush=True)
    print("[COMPARE] Comparing models...", flush=True)
    print("=" * 80, flush=True)
    
    # Load metrics
    if not os.path.exists(OLD_EVAL_PATH):
        print(f"[COMPARE] ⚠️ Old model metrics not found: {OLD_EVAL_PATH}", flush=True)
        print(f"[COMPARE] Assuming new model is better (first training)", flush=True)
        old_f1 = 0.0
        old_metrics = {}
    else:
        with open(OLD_EVAL_PATH, "r") as f:
            old_metrics = json.load(f)
        old_f1 = old_metrics["overall"]["f1"]
    
    if not os.path.exists(NEW_EVAL_PATH):
        print(f"[COMPARE] ❌ New model metrics not found: {NEW_EVAL_PATH}", flush=True)
        print(f"[COMPARE] Cannot proceed without evaluation results", flush=True)
        exit(1)
    
    with open(NEW_EVAL_PATH, "r") as f:
        new_metrics = json.load(f)
    new_f1 = new_metrics["overall"]["f1"]
    
    # Compare
    print(f"\n[COMPARE] Old model F1: {old_f1:.4f}", flush=True)
    print(f"[COMPARE] New model F1: {new_f1:.4f}", flush=True)
    print(f"[COMPARE] Improvement: {(new_f1 - old_f1):.4f}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = None

    # Chuẩn bị bản ghi lịch sử
    history_entry = {
        "timestamp": timestamp,
        "old_f1": old_f1,
        "new_f1": new_f1,
        "improvement": new_f1 - old_f1,
        "status": "SKIPPED_WORSE",
        "old_metrics_summary": {k:v for k,v in old_metrics.items() if k != 'overall'},
        "new_metrics_summary": {k:v for k,v in new_metrics.items() if k != 'overall'},
    }
    
    if new_f1 > old_f1:
        print(f"\n[COMPARE] ✅ New model is BETTER → Updating...", flush=True)
        history_entry["status"] = "UPDATED"

        # 1. Backup old model
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_count = count_backup_files(BACKUP_DIR) + 1 
        backup_path = os.path.join(BACKUP_DIR, f"best_absa_hardshare_{timestamp}_{backup_count:02d}.pt") # Thêm :02d để format thành 01, 02...

        if os.path.exists(OLD_MODEL_PATH):
            shutil.copy(OLD_MODEL_PATH, backup_path)
            print(f"[COMPARE] 📦 Backed up old model → {backup_path}", flush=True)
        
        # 2. Replace old model with new model (Sử dụng os.rename/os.remove để tránh lỗi quyền)
        try:
            # Ghi đè file metrics cũ = file metrics mới
            os.remove(OLD_EVAL_PATH) 
            os.rename(NEW_EVAL_PATH, OLD_EVAL_PATH)
            
            # Ghi đè file mô hình cũ = mô hình mới
            os.remove(OLD_MODEL_PATH) 
            os.rename(NEW_MODEL_PATH, OLD_MODEL_PATH)

            print(f"[COMPARE] ✅ Model updated (via rename): {OLD_MODEL_PATH}", flush=True)
            
        except PermissionError as pe:
            print(f"[COMPARE] ❌ CẬP NHẬT THẤT BẠI: PermissionError trong thao tác ghi đè (rename/remove): {pe}", flush=True)
        
        # 3. Signal for consumer restart (create flag file)
        flag_path = "/opt/airflow/models/MODEL_UPDATED"
        with open(flag_path, "w") as f:
            f.write(timestamp)
        print(f"[COMPARE] 🚩 Model update flag created: {flag_path}", flush=True)
        
    else:
        print(f"\n[COMPARE] ⚠️ New model is NOT better → Keeping old model", flush=True)
        print(f"[COMPARE] No update performed.", flush=True)
        history_entry["status"] = "SKIPPED_WORSE"
    
    # GHI LỊCH SỬ VÀO FILE JSONL
    if backup_path:
        history_entry["backup_path"] = backup_path
        
    try:
        with open(HISTORY_LOG_PATH, "a") as f: 
            f.write(json.dumps(history_entry) + "\n")
        print(f"[COMPARE] 📝 History logged to {HISTORY_LOG_PATH}", flush=True)
    except Exception as e:
        print(f"[COMPARE] ❌ Failed to write history log: {e}", flush=True)
    
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()


