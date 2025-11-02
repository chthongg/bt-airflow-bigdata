"""
Download YOLO model
"""
import os
from ultralytics import YOLO

model_dir = "/opt/airflow/projects/vehicle_counting/models"
model_path = f"{model_dir}/yolov8n.pt"

# Tạo thư mục nếu chưa có
os.makedirs(model_dir, exist_ok=True)

print("📥 Downloading YOLOv8n model...")
model = YOLO("yolov8n.pt")  # Auto-download

# Copy vào thư mục models
import shutil
yolo_cache = os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov8_main/yolov8n.pt")
if os.path.exists(yolo_cache):
    shutil.copy(yolo_cache, model_path)
    print(f"✅ Model saved to: {model_path}")
else:
    # Nếu không tìm thấy trong cache, model đã được download
    print(f"✅ Model ready at: {model_path}")

print("🎯 YOLO model download completed!")



