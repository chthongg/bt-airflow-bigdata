"""
SE363 - Vehicle Counting System
Producer 1: Video 1 → YOLO detection → Kafka
"""
import cv2
import json
import time
import base64
from ultralytics import YOLO
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# === CONFIGURATION ===
KAFKA_SERVER = "kafka-airflow:9092"
TOPIC = "vehicle-frames"
VIDEO_PATH = "/opt/airflow/projects/vehicle_counting/data/video1.mp4"
VIDEO_ID = "video_1"
MODEL_PATH = "/opt/airflow/models/yolov8n.pt"
FRAME_SKIP = 5

# --- Function to create Kafka topic ---
def create_kafka_topic(bootstrap_servers, topic_name):
    """Creates a Kafka topic if it does not already exist."""
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='producer_video1_admin'
        )
        
        topic_list = [NewTopic(name=topic_name, num_partitions=2, replication_factor=1)]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        print(f"Topic '{topic_name}' created successfully.")
        admin_client.close()
        
    except TopicAlreadyExistsError:
        print(f"Topic '{topic_name}' already exists. Skipping creation.")
    except Exception as e:
        print(f"Error creating topic '{topic_name}': {e}")

# === MAIN EXECUTION ===
def main():
    """Main function to set up, process video, and send to Kafka."""
    # 1. Create Kafka Topic
    create_kafka_topic(KAFKA_SERVER, TOPIC)

    # 2. Load YOLO Model
    print("Loading YOLO model...", flush=True)
    try:
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading YOLO model from {MODEL_PATH}: {e}", flush=True)
        return

    # 3. Initialize Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        max_request_size=10485760  # 10MB
    )
    print("Kafka Producer initialized.", flush=True)

    # 4. Process Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}", flush=True)
        return

    frame_count = 0
    sent_count = 0
    print(f"Processing video: {VIDEO_PATH} (sending every {FRAME_SKIP}th frame)", flush=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # YOLO detection
        results = model(frame, verbose=False)
        
        vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck', 3: 'motorbike'}
        detections = []
        vehicle_counts = {v: 0 for v in vehicle_classes.values()}
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in vehicle_classes:
                    vehicle_type = vehicle_classes[class_id]
                    vehicle_counts[vehicle_type] += 1
                    detections.append({
                        'type': vehicle_type,
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        # Encode a smaller frame for visualization
        small_frame = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create message payload
        message = {
            'video_id': VIDEO_ID,
            'frame_number': frame_count,
            'timestamp': time.time(),
            'vehicle_counts': vehicle_counts,
            'total_vehicles': sum(vehicle_counts.values()),
            'detections': detections,
            'frame_base64': frame_b64
        }
        
        # Send message to Kafka
        try:
            producer.send(TOPIC, value=message)
            sent_count += 1
            # Log progress every 10 frames
            if sent_count % 10 == 0:
                print(f"[{VIDEO_ID}] Sent {sent_count} frames | Latest: frame_{frame_count} with {sum(vehicle_counts.values())} vehicles", flush=True)
        except Exception as e:
            print(f"Error sending frame {frame_count}: {e}", flush=True)

    # Clean up
    cap.release()
    producer.flush()
    producer.close()
    print(f"Finished. Total frames sent: {sent_count}", flush=True)

if __name__ == "__main__":
    main()

