"""
Vehicle Counting System DAG

Orchestrates a real-time streaming pipeline for vehicle counting.
Architecture: 2 Producers (from video files) -> Kafka -> Spark Streaming -> PostgreSQL
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import subprocess
import psycopg2

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="vehicle_counting_system",
    default_args=default_args,
    description="Real-time vehicle counting with Kafka, Spark, and Airflow",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vehicle", "streaming", "yolo", "kafka", "spark"],
) as dag:

    # --- Task 1: Download YOLO model if it doesn't exist ---
    download_yolo = BashOperator(
        task_id="SETUP_download_yolov8_weights",
        bash_command="python /opt/airflow/projects/vehicle_counting/scripts/download_yolo.py",
    )

    # --- Task 2: Verify infrastructure is ready ---
    def verify_infrastructure():
        """Checks if Kafka and PostgreSQL are reachable."""
        print("Verifying infrastructure: Kafka, PostgreSQL.")
        
        # Check Kafka connection
        kafka_check = subprocess.run(["nc", "-zv", "kafka-airflow", "9092"], capture_output=True, text=True)
        if kafka_check.returncode != 0:
            raise ConnectionError(f"Kafka is not reachable: {kafka_check.stderr}")
        print("Kafka is reachable.")
        
        # Check PostgreSQL connection
        try:
            psycopg2.connect(host="postgres", database="airflow", user="airflow", password="airflow")
            print("PostgreSQL is reachable.")
        except Exception as e:
            raise ConnectionError(f"PostgreSQL connection failed: {e}")

    verify_infra = PythonOperator(
        task_id="SETUP_check_connection_Kafka_Postgres",
        python_callable=verify_infrastructure,
    )

    # --- Task 3: Cleanup old database records BEFORE pipeline starts ---
    def cleanup_old_data():
        """Truncates the vehicle_counts table to ensure clean state for this run."""
        print("Cleaning up old vehicle_counts records...", flush=True)
        try:
            conn = psycopg2.connect(
                host="postgres", 
                database="airflow", 
                user="airflow", 
                password="airflow",
                connect_timeout=10
            )
            cursor = conn.cursor()
            
            # Check row count before truncate
            cursor.execute("SELECT COUNT(*) FROM vehicle_counts;")
            count = cursor.fetchone()[0]
            print(f"Found {count} existing records. Truncating...", flush=True)
            
            # Use TRUNCATE with CASCADE to handle foreign keys
            cursor.execute("TRUNCATE TABLE vehicle_counts RESTART IDENTITY CASCADE;")
            conn.commit()
            
            cursor.close()
            conn.close()
            print(f"Deleted {count} old records. Database is clean.", flush=True)
        except Exception as e:
            print(f"Warning: Could not truncate table: {e}", flush=True)
            print("Table may not exist yet or is locked. Continuing...", flush=True)

    cleanup_database = PythonOperator(
        task_id="SETUP_clear_old_records",
        python_callable=cleanup_old_data,
    )

    # --- Task 4: Clean up old Spark checkpoints BEFORE pipeline starts ---
    cleanup_checkpoints = BashOperator(
        task_id="SETUP_clear_spark_checkpoint",
        bash_command="echo 'Removing old Spark checkpoints...'; rm -rf /opt/airflow/checkpoints/vehicle_counting_checkpoint || true;",
    )

    # --- Task 5: Start Spark Streaming Consumer ---
    # This task implements "KHỞI ĐỘNG" and "DỪNG" phases:
    # - Starts consumer (KHỞI ĐỘNG)
    # - Runs for 2 minutes (GIÁM SÁT happens during this time)
    # - Auto-stops via timeout (DỪNG)
    start_consumer = BashOperator(
        task_id="DEPLOY_consumer",
        bash_command='python /opt/airflow/projects/vehicle_counting/scripts/consumer_spark.py',
    )

    # --- Task 5a & 5b: Start Producers to send video frames to Kafka ---
    # These run in parallel. The logic to create the topic is now inside the producer scripts.
    start_producer1 = BashOperator(
        task_id="DEPLOY_producer_video1",
        bash_command="python /opt/airflow/projects/vehicle_counting/scripts/producer_video1.py",
        execution_timeout=timedelta(minutes=10),
    )
    
    start_producer2 = BashOperator(
        task_id="DEPLOY_producer_video2",
        bash_command="python /opt/airflow/projects/vehicle_counting/scripts/producer_video2.py",
        execution_timeout=timedelta(minutes=10),
    )

    # --- Task 6: Monitor the pipeline ---
    # This task implements "GIÁM SÁT" phase by verifying data was written to database
    def monitor_pipeline():
        """Monitors the pipeline by checking if data exists in PostgreSQL."""
        import time
        
        print("=== GIÁM SÁT: Verifying pipeline results ===")
        print("Monitoring pipeline: Verifying data in PostgreSQL...")
        
        # # Wait a bit for consumer to process some data
        # print("Waiting 10 seconds for consumer to start processing...")
        # time.sleep(10)
        
        try:
            conn = psycopg2.connect(host="postgres", database="airflow", user="airflow", password="airflow")
            cursor = conn.cursor()
            
            # Check multiple times
            for attempt in range(1, 4):
                cursor.execute("SELECT COUNT(*) FROM vehicle_counts;")
                count = cursor.fetchone()[0]
                print(f"[Attempt {attempt}/3] Total records found: {count}")
                
                if count > 0:
                    break
                    
                if attempt < 3:
                    print(f"No data yet. Waiting 20 more seconds...")
                    time.sleep(20)
            
            if count == 0:
                print("Warning: No data found in the database after monitoring period.")
            else:
                cursor.execute("SELECT video_id, COUNT(*), SUM(total_vehicles) FROM vehicle_counts GROUP BY video_id;")
                results = cursor.fetchall()
                for row in results:
                    print(f"  - Video '{row[0]}': {row[1]} frames processed, {row[2]} vehicles counted.")
                print("Monitoring successful: Data is being written correctly.")
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Pipeline monitoring check failed: {e}")
    
    monitor_task = PythonOperator(
        task_id="MONITOR_verify_data",
        python_callable=monitor_pipeline,
    )

    # --- Task 8: Final cleanup ---
    # This task implements "DỌN DẸP" phase by removing temporary files
    final_cleanup = BashOperator(
        task_id="CLEANUP_temp_files",
        bash_command='echo "=== DỌN DẸP: Removing temporary files ===" && rm -f /tmp/stop_vehicle_counting || true',
        trigger_rule="all_done",
    )


    # --- DAG Dependencies ---
    # Lifecycle: KHỞI ĐỘNG – GIÁM SÁT – DỪNG – DỌN DẸP
    
    # Phase 1: SETUP - Prepare clean environment (before KHỞI ĐỘNG)
    [download_yolo, verify_infra] >> cleanup_database >> cleanup_checkpoints

    # Phase 2: KHỞI ĐỘNG (DEPLOY) - Start streaming components in parallel
    cleanup_checkpoints >> [start_consumer, start_producer1, start_producer2]
    
    # Phase 3: GIÁM SÁT (MONITOR) - Verify data is being processed
    # (Runs after DEPLOY finishes, as consumer auto-stops after 2 min = DỪNG)
    [start_consumer, start_producer1, start_producer2] >> monitor_task
    
    # Phase 4: DỌN DẸP (CLEANUP) - Remove temporary files after pipeline stops
    monitor_task >> final_cleanup

