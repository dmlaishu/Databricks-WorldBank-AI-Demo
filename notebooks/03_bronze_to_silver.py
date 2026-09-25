# Databricks notebook source
from pyspark.sql import functions as F

CATALOG = "world_bank_ai"

bronze = spark.table(f"{CATALOG}.bronze.indicators_raw")

silver = (
    bronze
    .filter(F.col("value").isNotNull())
    .withColumn("year", F.col("year").cast("int"))
    .withColumn("value", F.col("value").cast("double"))
    .dropDuplicates(["country_code", "indicator_code", "year"])
)

(silver.write.format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.silver.indicators_clean"))

display(silver.orderBy("country_code", "indicator_code", F.desc("year")))
