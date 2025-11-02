#!/bin/bash
# SE363 - Run Consumer for Vehicle Counting

echo "🚀 Starting Vehicle Counting Consumer (Spark Streaming)..."

# Tìm spark-submit từ pyspark
SPARK_SUBMIT=$(python -c "from pyspark.find_spark_home import _find_spark_home; import os; print(os.path.join(_find_spark_home(), 'bin', 'spark-submit'))")

echo "📍 Using spark-submit: $SPARK_SUBMIT"

$SPARK_SUBMIT \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0,org.apache.kafka:kafka-clients:3.5.1 \
  /opt/airflow/projects/vehicle_counting/scripts/consumer_spark.py


