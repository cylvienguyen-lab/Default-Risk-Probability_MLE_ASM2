"""Ingest a feature source snapshot without cleaning source values."""

from pathlib import Path

import pyspark.sql.functions as F

from pipelines.utils.config import month, partition
from pipelines.utils.validation import require_columns, unique_keys


SOURCE_COLUMNS = {
    "attributes": ["Customer_ID", "Name", "Age", "SSN", "Occupation", "snapshot_date"],
    "financials": [
        "Customer_ID", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
        "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour",
        "Monthly_Balance", "snapshot_date",
    ],
    "clickstream": ["Customer_ID", "snapshot_date"] + [f"fe_{n}" for n in range(1, 21)],
}


def process_bronze_feature(snapshot_date_str: str, source_file_path: Path,
                           bronze_directory: Path, table_name: str, spark,
                           max_rows: int = 1000000):
    # 1. Validate source and requested snapshot
    source = Path(source_file_path).resolve()
    if table_name not in SOURCE_COLUMNS or not source.is_file():
        raise FileNotFoundError(f"Bronze {table_name}: missing source {source}")
    raw = spark.read.csv(str(source), header=True, inferSchema=True)
    require_columns(raw, SOURCE_COLUMNS[table_name], f"Bronze {table_name}")
    frame = raw.filter(F.to_date("snapshot_date") == F.lit(month(snapshot_date_str)))
    count = frame.count()
    if count == 0 or count > max_rows:
        raise ValueError(f"Bronze {table_name} {snapshot_date_str}: invalid row count {count}")
    unique_keys(frame, ["Customer_ID", "snapshot_date"], f"Bronze {table_name}")

    # 2. Write the bounded snapshot in the inherited single-CSV layout
    output = partition(Path(bronze_directory), f"bronze_{table_name}",
                       snapshot_date_str, "csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.toPandas().to_csv(output, index=False)
    return frame
