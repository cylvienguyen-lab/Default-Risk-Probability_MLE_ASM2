"""Ingest one LMS outcome snapshot."""

from pathlib import Path

import pyspark.sql.functions as F

from pipelines.utils.config import month, partition
from pipelines.utils.validation import require_columns, unique_keys


def process_bronze_table(snapshot_date_str: str, source_file_path: Path,
                         bronze_lms_directory: Path, spark,
                         max_rows: int = 1000000):
    # 1. Validate source, schema, and loan/snapshot grain
    source = Path(source_file_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bronze LMS: missing source {source}")
    raw = spark.read.csv(str(source), header=True, inferSchema=True)
    require_columns(raw, ["loan_id", "Customer_ID", "loan_start_date", "tenure",
                          "installment_num", "loan_amt", "due_amt", "paid_amt",
                          "overdue_amt", "balance", "snapshot_date"], "Bronze LMS")
    frame = raw.filter(F.to_date("snapshot_date") == F.lit(month(snapshot_date_str)))
    count = frame.count()
    if count == 0 or count > max_rows:
        raise ValueError(f"Bronze LMS {snapshot_date_str}: invalid row count {count}")
    unique_keys(frame, ["loan_id", "snapshot_date"], "Bronze LMS")

    # 2. Write the bounded snapshot in the inherited single-CSV layout
    output = partition(Path(bronze_lms_directory), "bronze_loan_daily",
                       snapshot_date_str, "csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.toPandas().to_csv(output, index=False)
    return frame
