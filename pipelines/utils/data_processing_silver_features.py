import os
from pathlib import Path
from datetime import datetime

import pyspark.sql.functions as F
from pipelines.utils.artifacts import write_spark_parquet_local

from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
)

from pipelines.utils.validation import unique_keys


def process_silver_attributes(
    snapshot_date_str,
    bronze_attributes_directory,
    silver_attributes_directory,
    spark,
):

    # validate the snapshot date
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    bronze_partition_name = (
        f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv"
    )
    bronze_filepath = os.path.join(
        bronze_attributes_directory,
        bronze_partition_name,
    )
    df = spark.read.csv(bronze_filepath, header=True, inferSchema=True)
    print("loaded from:", bronze_filepath, "row count:", df.count())

    # clean Age format: "39_" -> "39", "-5" remains "-5".
    age_text = F.regexp_replace(F.trim(col("Age")), r"_+$", "")
    parsed_age = F.when(
        age_text.rlike(r"^-?\d+$"),
        age_text.cast(IntegerType()),
    ).otherwise(F.lit(None).cast(IntegerType()))

    # retain plausible ages and mark invalid ages as missing
    df = df.withColumn(
        "Age",
        F.when(
            (parsed_age > 0) & (parsed_age < 150),
            parsed_age,
        ).otherwise(F.lit(None).cast(IntegerType())),
    )

    # normalize Occupation and convert unavailable values to Unknown
    occupation_text = F.trim(col("Occupation"))
    df = df.withColumn(
        "Occupation",
        F.when(
            col("Occupation").isNull()
            | (occupation_text == "")
            | occupation_text.rlike(r"^_+$"),
            F.lit("Unknown"),
        ).otherwise(occupation_text),
    )

    # enforce schema
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column_name, new_type in column_type_map.items():
        df = df.withColumn(column_name, col(column_name).cast(new_type))

    unique_keys(df, ["Customer_ID", "snapshot_date"], "Silver Attributes")
    if df.filter(F.col("snapshot_date").isNull() | F.col("Occupation").isNull() |
                 (F.col("snapshot_date") != F.lit(datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()))).limit(1).count():
        raise ValueError(f"Silver Attributes {snapshot_date_str}: invalid required fields")

    # save silver
    os.makedirs(silver_attributes_directory, exist_ok=True)
    silver_partition_name = (
        f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    silver_filepath = os.path.join(
        silver_attributes_directory,
        silver_partition_name,
    )
    write_spark_parquet_local(df, Path(silver_filepath))
    print("saved to:", silver_filepath)

    return df


def process_silver_financials(
    snapshot_date_str,
    bronze_financials_directory,
    silver_financials_directory,
    spark,
):

    # 1. Validate snapshot date and read Bronze
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    bronze_partition_name = (
        f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv"
    )
    bronze_filepath = os.path.join(
        bronze_financials_directory,
        bronze_partition_name,
    )

    df = spark.read.csv(
        bronze_filepath,
        header=True,
        inferSchema=True,
    )
    print("loaded from:", bronze_filepath, "row count:", df.count())

    # 2. Normalize numeric formats and cast datatypes
    df = (
        df
        .withColumn(
            "Annual_Income",
            F.regexp_replace(
                F.trim(col("Annual_Income")),
                r"_+$",
                "",
            ).cast(DoubleType()),
        )
        .withColumn(
            "Num_of_Loan",
            F.regexp_replace(
                F.trim(col("Num_of_Loan")),
                r"_+$",
                "",
            ).cast(IntegerType()),
        )
        .withColumn(
            "Num_of_Delayed_Payment",
            F.regexp_replace(
                F.trim(col("Num_of_Delayed_Payment")),
                r"_+$",
                "",
            ).cast(IntegerType()),
        )
        .withColumn(
            "Changed_Credit_Limit",
            F.when(
                F.trim(col("Changed_Credit_Limit")) == "_",
                F.lit(None),
            ).otherwise(
                F.trim(col("Changed_Credit_Limit")),
            ).cast(DoubleType()),
        )
        .withColumn(
            "Outstanding_Debt",
            F.regexp_replace(
                F.trim(col("Outstanding_Debt")),
                r"_+$",
                "",
            ).cast(DoubleType()),
        )
        .withColumn(
            "Amount_invested_monthly",
            F.when(
                F.trim(col("Amount_invested_monthly")) == "__10000__",
                F.lit(None),
            ).otherwise(
                F.trim(col("Amount_invested_monthly")),
            ).cast(DoubleType()),
        )
        .withColumn(
            "Monthly_Balance",
            F.when(
                F.trim(col("Monthly_Balance"))
                == "__-333333333333333333333333333__",
                F.lit(None),
            ).otherwise(
                F.trim(col("Monthly_Balance")),
            ).cast(DoubleType()),
        )
    )

    # 3. Create unusual discrete-value flag
    unusual_discrete_condition = (
        (col("Num_Bank_Accounts") > 11)
        | (col("Num_Credit_Card") > 11)
        | col("Interest_Rate").between(35, 99)
        | (col("Num_of_Delayed_Payment") > 28)
        | (col("Num_Credit_Inquiries") > 17)
    )

    df = df.withColumn(
        "unusual_discrete_value_flag",
        F.when(unusual_discrete_condition, 1)
        .otherwise(0)
        .cast(IntegerType()),
    )

    # 4. Apply confirmed numeric invalid rules
    invalid_numeric_rules = {
        "Num_Bank_Accounts": col("Num_Bank_Accounts") < 0,
        "Num_Credit_Card": col("Num_Credit_Card") < 0,
        "Interest_Rate": (
            (col("Interest_Rate") < 0)
            | (col("Interest_Rate") >= 100)
        ),
        "Num_of_Delayed_Payment": (
            col("Num_of_Delayed_Payment") < 0
        ),
        "Num_Credit_Inquiries": col("Num_Credit_Inquiries") < 0,
    }

    for column_name, invalid_condition in invalid_numeric_rules.items():
        df = df.withColumn(
            column_name,
            F.when(
                invalid_condition,
                F.lit(None),
            ).otherwise(col(column_name)),
        )

    # 5. Normalize simple categorical columns
    credit_mix_text = F.trim(col("Credit_Mix"))
    df = df.withColumn(
        "Credit_Mix",
        F.when(
            col("Credit_Mix").isNull()
            | (credit_mix_text == "")
            | (credit_mix_text == "_"),
            F.lit("Unknown"),
        ).otherwise(credit_mix_text),
    )

    payment_minimum_text = F.trim(col("Payment_of_Min_Amount"))
    df = df.withColumn(
        "Payment_of_Min_Amount",
        F.when(
            col("Payment_of_Min_Amount").isNull()
            | (payment_minimum_text == "")
            | (payment_minimum_text == "NM"),
            F.lit("Unknown"),
        ).otherwise(payment_minimum_text),
    )

    payment_behaviour_text = F.trim(col("Payment_Behaviour"))
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(
            col("Payment_Behaviour").isNull()
            | (payment_behaviour_text == "")
            | (payment_behaviour_text == "!@9#%8"),
            F.lit("Unknown"),
        ).otherwise(payment_behaviour_text),
    )

    # 6. Parse Credit_History_Age into months
    credit_history_text = F.trim(col("Credit_History_Age"))
    credit_history_pattern = r"^([0-9]+) Years and ([0-9]+) Months$"

    history_years = F.regexp_extract(
        credit_history_text,
        credit_history_pattern,
        1,
    ).cast(IntegerType())
    history_months = F.regexp_extract(
        credit_history_text,
        credit_history_pattern,
        2,
    ).cast(IntegerType())

    df = (
        df
        .withColumn(
            "Credit_History_Age_Months",
            F.when(
                credit_history_text.rlike(credit_history_pattern)
                & history_months.between(0, 11),
                history_years * 12 + history_months,
            ).otherwise(F.lit(None).cast(IntegerType())),
        )
        .drop("Credit_History_Age")
    )

    # 7. Parse Type_of_Loan into Loan_Type_Array
    loan_type_text = F.trim(col("Type_of_Loan"))
    normalized_loan_type_text = F.regexp_replace(
        F.regexp_replace(
            loan_type_text,
            ", and ",
            ", ",
        ),
        " and ",
        ", ",
    )
    parsed_loan_types = F.split(
        normalized_loan_type_text,
        r",\s*",
    )

    df = (
        df
        .withColumn(
            "Loan_Type_Array",
            F.when(
                col("Type_of_Loan").isNotNull()
                & (loan_type_text != ""),
                F.transform(
                    parsed_loan_types,
                    lambda loan_type: F.when(
                        F.trim(loan_type) == "Not Specified",
                        F.lit("Unknown"),
                    ).otherwise(F.trim(loan_type)),
                ),
            ),
        )
        .drop("Type_of_Loan")
    )

    # 8. Cross-check loan information
    loan_entry_count = F.size(col("Loan_Type_Array"))
    loan_type_missing = col("Loan_Type_Array").isNull()
    consistent_no_loan = (
        (col("Num_of_Loan") == 0)
        & loan_type_missing
        & (col("Total_EMI_per_month") == 0)
    )
    invalid_loan_count = col("Num_of_Loan") < 0
    unsupported_loan_count = (
        (col("Num_of_Loan") > 9)
        & (
            loan_type_missing
            | (loan_entry_count != col("Num_of_Loan"))
        )
    )
    zero_loan_conflict = (
        (col("Num_of_Loan") == 0)
        & (
            (~loan_type_missing)
            | (col("Total_EMI_per_month") > 0)
        )
    )
    positive_loan_conflict = (
        col("Num_of_Loan").between(1, 9)
        & (
            loan_type_missing
            | (loan_entry_count != col("Num_of_Loan"))
        )
    )

    loan_inconsistent_condition = (
        invalid_loan_count
        | unsupported_loan_count
        | zero_loan_conflict
        | positive_loan_conflict
    )

    df = (
        df
        .withColumn(
            "loan_info_inconsistent_flag",
            F.when(loan_inconsistent_condition, 1)
            .otherwise(0)
            .cast(IntegerType()),
        )
        .withColumn(
            "Loan_Type_Array",
            F.when(
                consistent_no_loan,
                F.array(F.lit("No Loan")),
            )
            .when(
                loan_type_missing,
                F.array(F.lit("Unknown")),
            )
            .otherwise(col("Loan_Type_Array")),
        )
        .withColumn(
            "Num_of_Loan",
            F.when(
                invalid_loan_count | unsupported_loan_count,
                F.lit(None).cast(IntegerType()),
            ).otherwise(col("Num_of_Loan")),
        )
    )

    # 9. Enforce final schema
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DoubleType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Loan_Type_Array": ArrayType(StringType()),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": DoubleType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Credit_History_Age_Months": IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DoubleType(),
        "snapshot_date": DateType(),
        "unusual_discrete_value_flag": IntegerType(),
        "loan_info_inconsistent_flag": IntegerType(),
    }

    for column_name, new_type in column_type_map.items():
        df = df.withColumn(column_name, col(column_name).cast(new_type))

    unique_keys(df, ["Customer_ID", "snapshot_date"], "Silver Financials")
    if df.filter(F.col("snapshot_date").isNull() | F.col("Annual_Income").isNull() |
                 (F.col("snapshot_date") != F.lit(datetime.strptime(snapshot_date_str, "%Y-%m-%d").date()))).limit(1).count():
        raise ValueError(f"Silver Financials {snapshot_date_str}: invalid required fields")

    # 10. Save Silver
    os.makedirs(silver_financials_directory, exist_ok=True)
    silver_partition_name = (
        f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    silver_filepath = os.path.join(
        silver_financials_directory,
        silver_partition_name,
    )

    write_spark_parquet_local(df, Path(silver_filepath))
    print("saved to:", silver_filepath)

    return df


def process_silver_clickstream(
    snapshot_date_str,
    bronze_clickstream_directory,
    silver_clickstream_directory,
    spark,
):

    # 1. Validate snapshot date and read Bronze
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    bronze_partition_name = (
        f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv"
    )
    bronze_filepath = os.path.join(
        bronze_clickstream_directory,
        bronze_partition_name,
    )

    df = spark.read.csv(
        bronze_filepath,
        header=True,
        inferSchema=True,
    )
    print("loaded from:", bronze_filepath, "row count:", df.count())

    # 2. Enforce schema while preserving all feature values
    feature_columns = [f"fe_{index}" for index in range(1, 21)]

    column_type_map = {
        "Customer_ID": StringType(),
        **{
            column_name: IntegerType()
            for column_name in feature_columns
        },
        "snapshot_date": DateType(),
    }

    for column_name, new_type in column_type_map.items():
        df = df.withColumn(column_name, col(column_name).cast(new_type))

    df = df.select(
        "Customer_ID",
        *feature_columns,
        "snapshot_date",
    )
    unique_keys(df, ["Customer_ID", "snapshot_date"], "Silver Clickstream")
    if df.filter(F.col("snapshot_date") != F.lit(datetime.strptime(snapshot_date_str, "%Y-%m-%d").date())).limit(1).count():
        raise ValueError(f"Silver Clickstream {snapshot_date_str}: wrong snapshot")

    # 3. Save Silver
    os.makedirs(silver_clickstream_directory, exist_ok=True)
    silver_partition_name = (
        f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    silver_filepath = os.path.join(
        silver_clickstream_directory,
        silver_partition_name,
    )

    write_spark_parquet_local(df, Path(silver_filepath))
    print("saved to:", silver_filepath)

    return df
