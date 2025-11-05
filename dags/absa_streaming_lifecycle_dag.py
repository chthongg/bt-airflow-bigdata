# ===========================================
# DAG: ABSA Streaming Lifecycle Orchestration (1-Hour-30-minutes Demo)
# ===========================================
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os, subprocess

# === Default parameters ===
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,                          # Thử lại tối đa 2 lần nếu lỗi
    "retry_delay": timedelta(minutes=2),   # Mỗi lần retry cách nhau 2 phút
}

# === DAG definition ===
with DAG(
        dag_id="absa_streaming_lifecycle_demo",
        default_args=default_args,
        description="Orchestrate Kafka–Spark–PostgreSQL streaming + Model Retraining (Demo: 100 messages)",
        schedule_interval=timedelta(hours=1, minutes=30),            # None -> xong demo đổi lại timedelta(hours=1) Chu kỳ 1 giờ 
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        dagrun_timeout=timedelta(minutes=60),            # Tăng lên 60 phút cho retraining
        tags=["absa", "streaming", "kafka", "spark", "retraining"],
) as dag:

    # === 0️⃣ Reset state: truncate table + clear checkpoint ===
    def reset_state():
        # BỔ SUNG CÁC MODULE THIẾU VÀO IMPORT: KafkaAdminClient và UnknownTopicOrPartitionError
        import psycopg2, shutil, subprocess, time
        from psycopg2 import sql
        from kafka.admin import KafkaAdminClient
        from kafka.errors import UnknownTopicOrPartitionError 
        

        # --- 1. Xóa Kafka Topic (Sử dụng Python API) ---
        TOPIC = "absa-reviews"
        KAFKA_SERVER = "kafka-airflow:9092"
        
        print(f"[Reset] Attempting to delete Kafka topic: {TOPIC} using Kafka Admin API...")
        try:
            admin = KafkaAdminClient(bootstrap_servers=KAFKA_SERVER, client_id="absa_admin_reset")
            
            try:
                admin.delete_topics(topics=[TOPIC], timeout_ms=5000)
                print(f"[Reset] Sent delete command for topic {TOPIC}. Waiting 5s...")
            except UnknownTopicOrPartitionError:
                print(f"[Reset] Topic {TOPIC} does not exist or already deleted.")
            
            admin.close()
            time.sleep(5) 
        except Exception as e:
            # Nếu lỗi kết nối Kafka, nó sẽ báo lỗi và task này vẫn hoàn thành
            print(f"[Reset] Error connecting to Kafka Admin Client to delete topic: {e}")

        # --- 2. Xóa Checkpoint Spark ---
        ckpt = "/opt/airflow/checkpoints/absa_streaming_checkpoint" 
        try:
            shutil.rmtree(ckpt, ignore_errors=True)
            print("[Reset] Removed checkpoint directory.")
        except Exception as e:
            print(f"[Reset] Failed to remove checkpoint: {e}")

        # --- 3. Truncate table PostgreSQL ---
        try:
            conn = psycopg2.connect(
                dbname="airflow", user="airflow", password="airflow", host="postgres", port=5432 #
            )
            conn.autocommit = True
            cur = conn.cursor()
            try:
                # Tạo bảng nếu chưa tồn tại
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
                cur.execute("TRUNCATE TABLE absa_results;")
                print("[Reset] Truncated table absa_results.")
            except Exception as e:
                print(f"[Reset] Skip truncate (maybe table missing): {e}")
            finally:
                cur.close(); conn.close()
        except Exception as e:
            print(f"[Reset] Cannot connect PostgreSQL to truncate: {e}")

    init_reset = PythonOperator(
        task_id="init_reset",
        python_callable=reset_state,
        trigger_rule="all_done",
    )

    # === 1️⃣ Khởi động Producer ===
    deploy_producer = BashOperator(
        task_id="deploy_producer",
        bash_command='bash -c "cd /opt/airflow && PYTHONUNBUFFERED=1 python -u projects/absa_streaming/scripts/producer.py"',
        retries=3,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=12),         # 100 msg + buffer
        trigger_rule="all_done",
    )

    # === 2️⃣ Khởi động Consumer ===
    deploy_consumer = BashOperator(
        task_id="deploy_consumer",
        # Chạy consumer bằng Python trực tiếp để log unbuffered; để lộ lỗi nếu có
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache timeout 600s python -u projects/absa_streaming/scripts/consumer_postgres_streaming.py"',
        retries=5,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=18),         # Inference + PostgreSQL write
        trigger_rule="all_done",
    )

    # === 3️⃣ Giám sát checkpoint ===
    def monitor_job():
        print("[Monitor] Checking streaming job checkpoint...")
        path = "/opt/airflow/checkpoints/absa_streaming_checkpoint"
        if os.path.exists(path):
            size = subprocess.check_output(["du", "-sh", path]).decode().split()[0]
            print(f"[Monitor] Checkpoint exists ({size}) → job running normally.")
        else:
            print("[Monitor] No checkpoint found. Possibly failed or cleaned.")

    monitor_stream = PythonOperator(
        task_id="monitor_stream",
        python_callable=monitor_job,
        trigger_rule="all_done",
    )

    # === 4️⃣ RETRAINING PIPELINE ===
    # 4.1 Chuẩn bị training data
    prepare_training_data = BashOperator(
        task_id="prepare_training_data",
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache PYTHONUNBUFFERED=1 timeout 5m python -u projects/absa_streaming/scripts/prepare_training_data.py"',
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    # 4.2 Train model mới
    train_new_model = BashOperator(
        task_id="train_new_model",
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache PYTHONUNBUFFERED=1 timeout 20m python -u projects/absa_streaming/scripts/train_model.py"',
        execution_timeout=timedelta(minutes=58),
        trigger_rule="all_done",
    )

    # 4.3 Evaluate model cũ
    evaluate_old_model = BashOperator(
        task_id="evaluate_old_model",
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache PYTHONUNBUFFERED=1 timeout 5m python -u projects/absa_streaming/scripts/evaluate_model.py /opt/airflow/models/best_absa_hardshare.pt"',
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    # 4.4 Evaluate model mới
    evaluate_new_model = BashOperator(
        task_id="evaluate_new_model",
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache PYTHONUNBUFFERED=1 timeout 5m python -u projects/absa_streaming/scripts/evaluate_model.py /opt/airflow/models/new_absa_model.pt"',
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    # 4.5 So sánh và cập nhật model nếu tốt hơn
    compare_and_update = BashOperator(
        task_id="compare_and_update",
        bash_command='bash -c "cd /opt/airflow && HF_HOME=/opt/airflow/models/hf-cache PYTHONUNBUFFERED=1 timeout 2m python -u projects/absa_streaming/scripts/compare_and_update_model.py"',
        execution_timeout=timedelta(minutes=2),
        trigger_rule="all_done",
    )

    # === 5️⃣ Dọn dẹp checkpoint ===
    cleanup_checkpoints = BashOperator(
        task_id="cleanup_checkpoints",
        bash_command=(
            "echo '[Cleanup] Removing old checkpoint...'; "
            "rm -rf /opt/airflow/checkpoints/absa_streaming_checkpoint || true; "
            "echo '[Cleanup] Done.'"
        ),
        trigger_rule="all_done",
    )

    # === Task dependency ===
    # Phase 1: Streaming
    init_reset >> [deploy_producer, deploy_consumer] >> monitor_stream

    # Phase 2: Retraining (song song chuẩn bị data và đánh giá model cũ)
    monitor_stream >> [prepare_training_data, evaluate_old_model]
    prepare_training_data >> train_new_model >> evaluate_new_model
    [evaluate_old_model, evaluate_new_model] >> compare_and_update

    # Phase 3: Cleanup
    compare_and_update >> cleanup_checkpoints
