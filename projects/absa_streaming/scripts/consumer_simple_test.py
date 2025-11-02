# Simple consumer test - không dùng model ABSA
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql import types as T
import sys

# Spark session
spark = (
    SparkSession.builder
    .appName("Kafka_Simple_Test")
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
            "org.postgresql:postgresql:42.6.0")
    .config("spark.sql.streaming.checkpointLocation", "/opt/airflow/checkpoints/simple_test")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Đọc từ Kafka
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka-airflow:9092")
    .option("subscribe", "absa-reviews")
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 10)
    .load()
)

# Parse JSON
review_schema = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("review", T.StringType())
])

df_parsed = df_stream.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), review_schema).alias("data")) \
    .select("data.id", "data.review")

# Ghi vào PostgreSQL
def write_batch(batch_df, batch_id):
    sys.stdout.reconfigure(encoding='utf-8')
    count = batch_df.count()
    
    if count == 0:
        print(f"[Batch {batch_id}] ⚠️ Không có dữ liệu")
        return
    
    print(f"\n[Batch {batch_id}] ✅ Nhận {count} reviews")
    
    # Hiển thị 3 dòng mẫu
    samples = batch_df.limit(3).collect()
    for row in samples:
        print(f"  ID: {row.id}, Review: {row.review[:50]}...")
    
    try:
        # Ghi vào PostgreSQL
        batch_df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://postgres:5432/airflow") \
            .option("dbtable", "simple_reviews") \
            .option("user", "airflow") \
            .option("password", "airflow") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()
        print(f"[Batch {batch_id}] ✅ Ghi PostgreSQL thành công!")
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Lỗi ghi PostgreSQL: {e}")

# Start streaming
query = df_parsed.writeStream \
    .foreachBatch(write_batch) \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

print("🚀 Simple streaming test started...")
query.awaitTermination()



