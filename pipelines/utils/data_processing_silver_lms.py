"""Preserve the inherited monthly DPD approximation for valid LMS rows."""

from pathlib import Path

import pyspark.sql.functions as F
from pipelines.utils.artifacts import write_spark_parquet_local
from pyspark.sql.types import DateType, FloatType, IntegerType, StringType

from pipelines.utils.config import month, partition
from pipelines.utils.validation import require_columns, unique_keys


def process_silver_table(snapshot_date_str: str, bronze_lms_directory: Path,
                         silver_loan_daily_directory: Path, spark):
    # 1. Read and validate Bronze
    source = partition(Path(bronze_lms_directory), "bronze_loan_daily",
                       snapshot_date_str, "csv")
    frame = spark.read.csv(str(source), header=True, inferSchema=True)
    require_columns(frame, ["loan_id", "Customer_ID", "loan_start_date", "tenure",
                            "installment_num", "loan_amt", "due_amt", "paid_amt",
                            "overdue_amt", "balance", "snapshot_date"], "Silver LMS")
    types = {"loan_id": StringType(), "Customer_ID": StringType(),
             "loan_start_date": DateType(), "tenure": IntegerType(),
             "installment_num": IntegerType(), "loan_amt": FloatType(),
             "due_amt": FloatType(), "paid_amt": FloatType(),
             "overdue_amt": FloatType(), "balance": FloatType(),
             "snapshot_date": DateType()}
    for name, data_type in types.items():
        frame = frame.withColumn(name, F.col(name).cast(data_type))
    unique_keys(frame, ["loan_id", "snapshot_date"], "Silver LMS")
    amount_columns = ("loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance")
    invalid_amount = F.lit(False)
    for name in amount_columns:
        invalid_amount = (invalid_amount | F.col(name).isNull() | F.isnan(name) |
                          (F.abs(F.col(name)) == float("inf")) | (F.col(name) < 0))
    invalid = (F.col("Customer_ID").isNull() | F.col("installment_num").isNull() |
               (F.col("installment_num") < 0) | F.col("loan_start_date").isNull() |
               (F.col("loan_amt") <= 0) | invalid_amount |
               ((F.col("due_amt") == 0) & (F.col("overdue_amt") > 0)) |
               ((F.col("installment_num") > 0) & (F.col("due_amt") <= 0)))
    if frame.filter(invalid | (F.col("snapshot_date") != F.lit(month(snapshot_date_str)))).limit(1).count():
        raise ValueError(f"Silver LMS {snapshot_date_str}: invalid amounts, keys, or snapshot")

    # 2. Preserve the existing DPD formula, with explicit valid MOB-0 zeros
    frame = frame.withColumn("mob", F.col("installment_num"))
    frame = frame.withColumn("installments_missed", F.when(
        (F.col("due_amt") == 0) & (F.col("overdue_amt") == 0), F.lit(0))
        .otherwise(F.ceil(F.col("overdue_amt") / F.col("due_amt"))).cast(IntegerType()))
    frame = frame.withColumn("first_missed_date", F.when(
        F.col("installments_missed") > 0,
        F.add_months(F.col("snapshot_date"), -F.col("installments_missed"))).cast(DateType()))
    frame = frame.withColumn("dpd", F.when(
        F.col("overdue_amt") > 0,
        F.datediff(F.col("snapshot_date"), F.col("first_missed_date")))
        .otherwise(0).cast(IntegerType()))
    if frame.filter(F.col("dpd").isNull() | (F.col("dpd") < 0)).limit(1).count():
        raise ValueError(f"Silver LMS {snapshot_date_str}: invalid inferred DPD")

    # 3. Write Silver partition
    output = partition(Path(silver_loan_daily_directory), "silver_loan_daily",
                       snapshot_date_str, "parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_spark_parquet_local(frame, output)
    return frame
