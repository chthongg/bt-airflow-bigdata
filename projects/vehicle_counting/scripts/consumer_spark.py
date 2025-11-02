"""
SE363 - Vehicle Counting System
Consumer: Reads from Kafka topic "vehicle-frames" (written by 2 parallel producers)
         → Spark Structured Streaming → PostgreSQL
"""
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import from_json, col
import sys, json
import time
import os

def main():
    """Main function to run the Spark Streaming job."""
    print("🚀 Vehicle counting streaming started...")

    # === 1. SPARK SESSION ===
    spark = (
        SparkSession.builder
        .appName("Vehicle_Counting_Streaming")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
                "org.postgresql:postgresql:42.6.0,"
                "org.apache.kafka:kafka-clients:3.5.1")
        .config("spark.executor.instances", "1")
        .config("spark.executor.cores", "1")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.streaming.checkpointLocation", 
                "/opt/airflow/checkpoints/vehicle_counting_checkpoint")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # === 2. ĐỌC TỪ KAFKA ===
    df_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "kafka-airflow:9092")
        .option("subscribe", "vehicle-frames")
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 50)  # Xử lý 50 frames/batch
        .load()
    )

    # === 3. PARSE JSON ===
    schema = T.StructType([
        T.StructField("video_id", T.StringType()),
        T.StructField("frame_number", T.IntegerType()),
        T.StructField("timestamp", T.DoubleType()),
        T.StructField("vehicle_counts", T.MapType(T.StringType(), T.IntegerType())),
        T.StructField("total_vehicles", T.IntegerType()),
        T.StructField("detections", T.ArrayType(T.MapType(T.StringType(), T.StringType()))),
        T.StructField("frame_base64", T.StringType())
    ])

    df_parsed = (
        df_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), schema).alias("data"))
        .select("data.*")
    )

    # === 4. XỬ LÝ DỮ LIỆU ===
    # Tách vehicle_counts thành các cột riêng
    df_processed = (
        df_parsed
        .withColumn("car_count", col("vehicle_counts").getItem("car"))
        .withColumn("bus_count", col("vehicle_counts").getItem("bus"))
        .withColumn("truck_count", col("vehicle_counts").getItem("truck"))
        .withColumn("motorbike_count", col("vehicle_counts").getItem("motorbike"))
        .withColumn("processed_at", F.current_timestamp())
        .select(
            "video_id",
            "frame_number",
            F.to_timestamp(F.from_unixtime("timestamp")).alias("capture_time"),
            "car_count",
            "bus_count", 
            "truck_count",
            "motorbike_count",
            "total_vehicles",
            "processed_at"
        )
    )

    # === 5. GHI VÀO POSTGRESQL ===
    def write_to_postgres(batch_df, batch_id):
        sys.stdout.reconfigure(encoding='utf-8')
        count = batch_df.count()
        
        if count == 0:
            print(f"[Batch {batch_id}] ⚠️ No data")
            return
        
        print(f"\n[Batch {batch_id}] Processing {count} frames")
        
        # Hiển thị thống kê
        stats = batch_df.agg(
            F.sum("total_vehicles").alias("total"),
            F.sum("car_count").alias("cars"),
            F.sum("bus_count").alias("buses"),
            F.sum("truck_count").alias("trucks"),
            F.sum("motorbike_count").alias("motorbikes")
        ).collect()[0]
        
        print(f"  📊 Total vehicles: {stats['total']}")
        print(f"     - Cars: {stats['cars']}")
        print(f"     - Buses: {stats['buses']}")
        print(f"     - Trucks: {stats['trucks']}")
        print(f"     - Motorbikes: {stats['motorbikes']}")
        
        try:
            # Deduplicate based on video_id + frame_number before writing
            batch_df_dedup = batch_df.dropDuplicates(["video_id", "frame_number"])
            dedup_count = batch_df_dedup.count()
            
            if dedup_count < count:
                print(f"  ⚠️ Removed {count - dedup_count} duplicates")
            
            # Ghi vào PostgreSQL
            batch_df_dedup.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://postgres:5432/airflow") \
                .option("dbtable", "vehicle_counts") \
                .option("user", "airflow") \
                .option("password", "airflow") \
                .option("driver", "org.postgresql.Driver") \
                .mode("append") \
                .save()
            
            print(f"[Batch {batch_id}] ✅ Saved {dedup_count} unique records to PostgreSQL")
            
        except Exception as e:
            print(f"[Batch {batch_id}] ⚠️ PostgreSQL error: {e}")
            # Fallback: print ra console
            batch_df.show(5, truncate=False)

    # === 6. START STREAMING ===
    query = (
        df_processed.writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .option("checkpointLocation", "/opt/airflow/checkpoints/vehicle_counting_checkpoint")
        .start()
    )

    # Timeout-based shutdown: Run for maximum 2 minutes
    import time
    timeout_seconds = 2 * 60  # 2 minutes
    start_time = time.time()
    
    print(f"Consumer will run for maximum {timeout_seconds/60} minutes...", flush=True)
    
    while (time.time() - start_time) < timeout_seconds:
        elapsed = int(time.time() - start_time)
        remaining = int(timeout_seconds - elapsed)
        print(f"Streaming active... Elapsed: {elapsed}s, Remaining: {remaining}s", flush=True)
        query.awaitTermination(60)  # Check every 60 seconds

    print("Timeout reached. Stopping the streaming query gracefully...", flush=True)
    query.stop()
    print("Streaming query stopped.", flush=True)

if __name__ == "__main__":
    main()

