"""Create the unchanged five-column MOB-6 Gold Label schema."""

from pathlib import Path

import pyspark.sql.functions as F
from pipelines.utils.artifacts import write_spark_parquet_local

from pipelines.utils.artifacts import read_spark_parquet_local
from pipelines.utils.config import month, partition
from pipelines.utils.validation import validate_label_partition, unique_keys


def process_labels_gold_table(snapshot_date_str: str, silver_loan_daily_directory: Path,
                              gold_label_store_directory: Path, spark,
                              dpd: int = 30, mob: int = 6):
    # 1. Load and validate MOB-6 source records
    source = partition(Path(silver_loan_daily_directory), "silver_loan_daily",
                       snapshot_date_str, "parquet")
    frame = read_spark_parquet_local(spark, source).filter(F.col("mob") == mob)
    unique_keys(frame, ["loan_id", "snapshot_date"], "Gold label source")
    unique_keys(frame, ["Customer_ID", "snapshot_date"], "Gold label source")
    if frame.filter(F.col("dpd").isNull() | (F.col("dpd") < 0) |
                    (F.col("snapshot_date") != F.lit(month(snapshot_date_str)))).limit(1).count():
        raise ValueError(f"Gold labels {snapshot_date_str}: invalid source DPD or snapshot")

    # 2. Apply the frozen target and validate output
    frame = frame.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0))
    frame = frame.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob"))
    frame = frame.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    validate_label_partition(frame, snapshot_date_str, f"{dpd}dpd_{mob}mob")

    # 3. Write one snapshot, including a valid empty MOB-6 partition
    output = partition(Path(gold_label_store_directory), "gold_label_store",
                       snapshot_date_str, "parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_spark_parquet_local(frame, output)
    return frame
