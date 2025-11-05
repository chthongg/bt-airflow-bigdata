# SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn
# Khoa Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
# HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

# consumer_postgres_streaming.py
# ======================================
# Consumer đọc dữ liệu từ Kafka topic "absa-reviews"
# → chạy inference mô hình ABSA (.pt)
# → ghi kết quả vào PostgreSQL
# → Airflow sẽ giám sát và khởi động lại khi job bị dừng.

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import from_json, col
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as tF
from transformers import AutoTokenizer, AutoModel
import random, time, os, sys, json

# === 1. Spark session với Kafka connector ===
scala_version = "2.12"
spark_version = "3.5.1"

spark = (
    SparkSession.builder
    .appName("Kafka_ABSA_Postgres")
    .config(
        "spark.jars.packages",
        f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version},"
        f"org.postgresql:postgresql:42.6.0,"
        f"org.apache.kafka:kafka-clients:3.5.1"
    )
    .config("spark.executor.instances", "1")  # Giới hạn 1 executor
    .config("spark.executor.cores", "1")      # Giới hạn 1 core
    .config("spark.driver.maxResultSize", "4g")  # Giới hạn kết quả
    .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/absa_streaming_checkpoint")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Không yêu cầu pyarrow
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# === 2. Đọc dữ liệu streaming từ Kafka ===
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka-airflow:9092")
    .option("subscribe", "absa-reviews")
    .option("startingOffsets", "earliest")  # Đọc từ đầu topic nếu chưa có checkpoint
    .option("failOnDataLoss", "false")
    .option("maxOffsetsPerTrigger", 5)  # Batch nhỏ hơn để tránh OOM
    .load()
)

df_text = df_stream.selectExpr("CAST(value AS STRING) as Review")

# === 3. Định nghĩa mô hình ABSA ===
ASPECTS = ["price","shipping","outlook","quality","size","shop_service","general","others"]
MODEL_NAME = "xlm-roberta-base"
TOKENIZER_LOCAL_DIR = "/opt/airflow/models/hf-cache/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
BACKBONE_LOCAL_DIR = TOKENIZER_LOCAL_DIR  # dùng cùng thư mục nếu đã được tải sẵn
MODEL_PATH = "/opt/airflow/models/best_absa_hardshare.pt"
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model, _tokenizer = None, None
_model_mtime = 0.0

class ABSAModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
        super().__init__()
        # Ưu tiên backbone local nếu có để tránh tải mạng lần đầu
        src = BACKBONE_LOCAL_DIR if os.path.isdir(BACKBONE_LOCAL_DIR) else model_name
        try:
            if src == BACKBONE_LOCAL_DIR:
                print(f"[ABSA] Loading backbone locally from {BACKBONE_LOCAL_DIR}", flush=True)
                self.backbone = AutoModel.from_pretrained(src, local_files_only=True)
            else:
                print(f"[ABSA] Downloading backbone {model_name} (first run may take time)", flush=True)
                self.backbone = AutoModel.from_pretrained(src)
        except Exception as e:
            print(f"[ABSA] Failed to load backbone from '{src}': {e}", flush=True)
            raise
        H = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_m = nn.Linear(H, num_aspects)
        self.head_s = nn.Linear(H, num_aspects * 3)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_m(h_cls), self.head_s(h_cls).view(-1, len(ASPECTS), 3)

# === Giải mã Review JSON thành text tiếng Việt trước khi stream ===
review_schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("review", T.StringType())
])
df_final = df_text.withColumn("text", from_json(col("Review"), review_schema).getField("review")).select("text")

# === Inference on driver within foreachBatch to avoid UDF worker crashes ===
def _load_model_once():
    global _model, _tokenizer, _model_mtime

    # 1. Lấy thời gian sửa đổi hiện tại của file mô hình
    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ABSA] ERROR: Model file not found at {MODEL_PATH}", flush=True)
        return

    # 2. Logic Hot-Reload: Tải lần đầu HOẶC nếu thời gian sửa đổi đã thay đổi
    if _model is None or current_mtime > _model_mtime:
        
        if _model is not None:
            print(f"[ABSA HOT-RELOAD] Mô hình đã được cập nhật (mtime: {_model_mtime} -> {current_mtime}), TẢI LẠI!", flush=True)
            
        print("[ABSA] Loading tokenizer/model…", flush=True)

        if os.path.isdir(TOKENIZER_LOCAL_DIR):
            print(f"[ABSA] Using local tokenizer at {TOKENIZER_LOCAL_DIR}", flush=True)
            _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_DIR, use_fast=True, local_files_only=True)
        else:
            print(f"[ABSA] Using remote tokenizer {MODEL_NAME}", flush=True)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        m = ABSAModel()
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state, strict=False)
        m.to(DEVICE).eval()
        _model = m

        # 4. Cập nhật thời gian sửa đổi mới
        _model_mtime = current_mtime
        print("[ABSA] Tokenizer/model ready.", flush=True)
    else:
        # Mô hình đã tải và chưa thay đổi
        pass

def _infer_batch(texts):
    _load_model_once()
    SENTIMENTS = ["POS", "NEU", "NEG"]
    outputs = []
    for t in texts:
        t = t or ""
        enc = _tokenizer(t, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            _, logits_s = _model(enc["input_ids"], enc["attention_mask"])
            probs = tF.softmax(logits_s, dim=-1)[0].detach().cpu().numpy().tolist()
        row = {"text": t}
        for i, asp in enumerate(ASPECTS):
            idx = int(max(range(3), key=lambda j: probs[i][j]))
            row[asp] = SENTIMENTS[idx]
        outputs.append(row)
    return outputs

# === 5. Ghi kết quả vào PostgreSQL (chuẩn UTF-8, log đầy đủ, xử lý lỗi an toàn) ===
# Auto-stop tracking
import threading
_last_batch_time = [None]  # Sẽ được set lần đầu khi có batch non-empty
_stop_event = threading.Event()
_batch_in_progress = [False]  # Tránh stop khi đang xử lý batch

def write_to_postgres(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    pdf = batch_df.select("text").toPandas()
    if pdf.empty:
        print(f"[Batch {batch_id}] ⚠️ Không có dữ liệu mới.")
        return

    # Đánh dấu bắt đầu batch để tránh stop giữa chừng
    _batch_in_progress[0] = True
    try:
        # Cập nhật thời gian batch cuối (chỉ khi có data)
        _last_batch_time[0] = time.time()

        results = _infer_batch(pdf["text"].tolist())
        out_df = pd.DataFrame(results)

        print(f"\n[Batch {batch_id}] Nhận {len(out_df)} dòng, hiển thị 5 dòng đầu:")
        print(json.dumps(out_df.head(5).to_dict(orient="records"), ensure_ascii=False, indent=2))

        try:
            cols = ["text"] + ASPECTS
            values = [tuple(row[c] for c in cols) for _, row in out_df.iterrows()]
            with psycopg2.connect(dbname="airflow", user="airflow", password="airflow", host="postgres", port=5432) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS absa_results (
                            text TEXT,
                            price TEXT,
                            shipping TEXT,
                            outlook TEXT,
                            quality TEXT,
                            size TEXT,
                            shop_service TEXT,
                            general TEXT,
                            others TEXT
                        );
                        """
                    )
                    execute_values(
                        cur,
                        f"INSERT INTO absa_results ({', '.join(cols)}) VALUES %s",
                        values,
                    )
            print(f"[Batch {batch_id}] ✅ Ghi PostgreSQL thành công ({len(out_df)} dòng).")
        except Exception as e:
            print(f"[Batch {batch_id}] ⚠️ Không thể ghi vào PostgreSQL: {e}")
    finally:
        # Đánh dấu kết thúc batch
        _batch_in_progress[0] = False

def monitor_inactivity(query_obj, timeout_seconds=120):
    """Dừng streaming query nếu không có batch mới trong timeout_seconds"""
    while not _stop_event.is_set():
        time.sleep(10)
        # Chỉ kiểm tra nếu đã có batch non-empty
        if _last_batch_time[0] is not None:
            elapsed = time.time() - _last_batch_time[0]
            if elapsed > timeout_seconds:
                # Tránh stop khi batch đang xử lý để không gây Py4J lỗi kết thúc
                if not _batch_in_progress[0]:
                    print(f"\n[AUTO-STOP] Không có batch mới trong {timeout_seconds}s → Dừng consumer.")
                    query_obj.stop()
                    _stop_event.set()
                    break
                else:
                    # Đợi thêm vòng sau khi batch xong
                    print(f"[AUTO-STOP] Chờ batch hiện tại kết thúc rồi mới dừng…")

# === 6. Bắt đầu stream ===
query = (
    df_final.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
)

# Khởi động monitor thread để auto-stop sau 120s không có batch mới
monitor_thread = threading.Thread(target=monitor_inactivity, args=(query, 120), daemon=True)
monitor_thread.start()

print("🚀 Streaming job starting — chuẩn bị lắng nghe dữ liệu từ Kafka...")
print("[AUTO-STOP] Sẽ tự động dừng sau 120s không có batch mới (kể từ batch đầu tiên).")
try:
    query.awaitTermination()
    print("✅ Consumer đã dừng (do auto-stop hoặc signal).")
except Exception as e:
    # Một số trường hợp Py4J ném lỗi khi stop giữa các callback; log rồi kết thúc êm
    msg = str(e)
    if "Py4J" in msg or "py4j" in msg:
        print(f"⚠️ Ignored Py4J termination noise: {e}")
    else:
        print(f"❌ Streaming job failed: {e}")
        raise

