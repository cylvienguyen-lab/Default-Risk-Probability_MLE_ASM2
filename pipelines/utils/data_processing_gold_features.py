import os
from pathlib import Path
from datetime import datetime

from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
from pipelines.utils.artifacts import read_spark_parquet_local, write_spark_parquet_local

from pipelines.utils.validation import unique_keys


GOLD_IDENTIFIER_COLUMNS = [
    "Customer_ID",
    "application_date",
]

GOLD_ATTRIBUTE_COLUMNS = [
    "Age",
    "Occupation",
]

GOLD_FINANCIAL_NUMERIC_COLUMNS = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

GOLD_FINANCIAL_CATEGORICAL_COLUMNS = [
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

GOLD_QUALITY_FLAG_COLUMNS = [
    "unusual_discrete_value_flag",
    "loan_info_inconsistent_flag",
]

FINANCIAL_RATIO_FEATURES = [
    "debt_to_annual_income_ratio",
    "emi_to_monthly_salary_ratio",
    "investment_to_monthly_salary_ratio",
    "balance_to_monthly_salary_ratio",
]

FINANCIAL_LOG_FEATURES = [
    "log_annual_income",
    "log_total_emi_per_month",
    "log_amount_invested_monthly",
    "log_emi_to_monthly_salary_ratio",
]

KNOWN_LOAN_TYPE_FEATURE_MAP = {
    "Auto Loan": "auto_loan_count",
    "Credit-Builder Loan": "credit_builder_loan_count",
    "Debt Consolidation Loan": "debt_consolidation_loan_count",
    "Home Equity Loan": "home_equity_loan_count",
    "Mortgage Loan": "mortgage_loan_count",
    "Payday Loan": "payday_loan_count",
    "Personal Loan": "personal_loan_count",
    "Student Loan": "student_loan_count",
}

LOAN_PORTFOLIO_FEATURES = list(
    KNOWN_LOAN_TYPE_FEATURE_MAP.values()
) + [
    "unknown_loan_type_count",
    "no_loan_flag",
    "loan_type_diversity",
]

PRE_CLICKSTREAM_GOLD_COLUMNS = (
    GOLD_IDENTIFIER_COLUMNS
    + GOLD_ATTRIBUTE_COLUMNS
    + GOLD_FINANCIAL_NUMERIC_COLUMNS
    + GOLD_FINANCIAL_CATEGORICAL_COLUMNS
    + GOLD_QUALITY_FLAG_COLUMNS
    + FINANCIAL_RATIO_FEATURES
    + FINANCIAL_LOG_FEATURES
    + LOAN_PORTFOLIO_FEATURES
)

CLICKSTREAM_FEATURE_COLUMNS = [
    f"fe_{index}"
    for index in range(1, 21)
]

CLICKSTREAM_AGGREGATE_COLUMNS = [
    f"fe_{feature_index}_{statistic}_3m"
    for feature_index in range(1, 21)
    for statistic in [
        "latest",
        "mean",
        "std",
    ]
]

CLICKSTREAM_AVAILABILITY_COLUMNS = [
    "clickstream_months_available",
    "clickstream_history_available",
]

FINAL_GOLD_COLUMNS = (
    PRE_CLICKSTREAM_GOLD_COLUMNS
    + CLICKSTREAM_AGGREGATE_COLUMNS
    + CLICKSTREAM_AVAILABILITY_COLUMNS
)


def _safe_ratio(
    numerator_column,
    denominator_column,
):
    return F.when(
        F.col(denominator_column) > 0,
        (
            F.col(numerator_column)
            / F.col(denominator_column)
        ),
    ).otherwise(
        F.lit(None).cast("double"),
    )


def _add_financial_features(df):
    df = (
        df
        .withColumn(
            "debt_to_annual_income_ratio",
            _safe_ratio(
                "Outstanding_Debt",
                "Annual_Income",
            ),
        )
        .withColumn(
            "emi_to_monthly_salary_ratio",
            _safe_ratio(
                "Total_EMI_per_month",
                "Monthly_Inhand_Salary",
            ),
        )
        .withColumn(
            "investment_to_monthly_salary_ratio",
            _safe_ratio(
                "Amount_invested_monthly",
                "Monthly_Inhand_Salary",
            ),
        )
        .withColumn(
            "balance_to_monthly_salary_ratio",
            _safe_ratio(
                "Monthly_Balance",
                "Monthly_Inhand_Salary",
            ),
        )
        .withColumn(
            "log_annual_income",
            F.log1p(F.col("Annual_Income")),
        )
        .withColumn(
            "log_total_emi_per_month",
            F.log1p(F.col("Total_EMI_per_month")),
        )
        .withColumn(
            "log_amount_invested_monthly",
            F.log1p(F.col("Amount_invested_monthly")),
        )
        .withColumn(
            "log_emi_to_monthly_salary_ratio",
            F.log1p(
                F.col("emi_to_monthly_salary_ratio"),
            ),
        )
    )

    for loan_type, feature_name in (
        KNOWN_LOAN_TYPE_FEATURE_MAP.items()
    ):
        df = df.withColumn(
            feature_name,
            F.size(
                F.filter(
                    F.col("Loan_Type_Array"),
                    lambda item: item == F.lit(loan_type),
                ),
            ),
        )

    known_loan_types = list(
        KNOWN_LOAN_TYPE_FEATURE_MAP.keys()
    )

    df = (
        df
        .withColumn(
            "unknown_loan_type_count",
            F.size(
                F.filter(
                    F.col("Loan_Type_Array"),
                    lambda item: item == F.lit("Unknown"),
                ),
            ),
        )
        .withColumn(
            "no_loan_flag",
            F.when(
                F.array_contains(
                    F.col("Loan_Type_Array"),
                    "No Loan",
                ),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "loan_type_diversity",
            F.size(
                F.array_distinct(
                    F.filter(
                        F.col("Loan_Type_Array"),
                        lambda item: item.isin(
                            *known_loan_types
                        ),
                    ),
                ),
            ),
        )
    )

    return df.select(*PRE_CLICKSTREAM_GOLD_COLUMNS)


def _aggregate_clickstream(
    application_base_df,
    clickstream_paths,
    spark,
):
    if not clickstream_paths:
        return None

    clickstream_df = read_spark_parquet_local(spark, *map(Path, clickstream_paths))
    unique_keys(clickstream_df, ["Customer_ID", "snapshot_date"], "Silver Clickstream")

    application_keys_df = application_base_df.select(
        "Customer_ID",
        "application_date",
    )

    clickstream_window_df = (
        clickstream_df
        .join(
            application_keys_df,
            on="Customer_ID",
            how="inner",
        )
        .filter(
            (
                F.col("snapshot_date")
                >= F.add_months(
                    F.col("application_date"),
                    -3,
                )
            )
            & (
                F.col("snapshot_date")
                < F.col("application_date")
            )
        )
    )

    aggregation_expressions = []

    for column_name in CLICKSTREAM_FEATURE_COLUMNS:
        aggregation_expressions.extend([
            F.max_by(
                F.col(column_name),
                F.col("snapshot_date"),
            ).alias(
                f"{column_name}_latest_3m"
            ),
            F.avg(
                F.col(column_name)
            ).alias(
                f"{column_name}_mean_3m"
            ),
            F.stddev_pop(
                F.col(column_name)
            ).alias(
                f"{column_name}_std_3m"
            ),
        ])

    return (
        clickstream_window_df
        .groupBy(
            "Customer_ID",
            "application_date",
        )
        .agg(
            F.countDistinct(
                "snapshot_date"
            ).alias(
                "clickstream_months_available"
            ),
            *aggregation_expressions,
        )
        .withColumn(
            "clickstream_history_available",
            F.lit(1),
        )
    )


def _add_empty_clickstream_features(df):
    for column_name in CLICKSTREAM_AGGREGATE_COLUMNS:
        output_type = (
            "int"
            if column_name.endswith("_latest_3m")
            else "double"
        )

        df = df.withColumn(
            column_name,
            F.lit(None).cast(output_type),
        )

    return (
        df
        .withColumn(
            "clickstream_months_available",
            F.lit(0).cast("int"),
        )
        .withColumn(
            "clickstream_history_available",
            F.lit(0).cast("int"),
        )
    )


def process_gold_feature_store(
    application_date_str,
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark,
):
    # 1. Validate application date and build required paths
    application_date = datetime.strptime(
        application_date_str,
        "%Y-%m-%d",
    )
    date_suffix = application_date_str.replace("-", "_")

    attributes_path = os.path.join(
        silver_attributes_directory,
        f"silver_attributes_{date_suffix}.parquet",
    )
    financials_path = os.path.join(
        silver_financials_directory,
        f"silver_financials_{date_suffix}.parquet",
    )

    for required_path in [
        attributes_path,
        financials_path,
    ]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"Required Silver partition not found: "
                f"{required_path}"
            )

    # 2. Read matching Silver application partitions
    attributes_df = read_spark_parquet_local(spark, Path(attributes_path))
    financials_df = read_spark_parquet_local(spark, Path(financials_path))
    unique_keys(attributes_df, ["Customer_ID", "snapshot_date"], "Silver Attributes")
    unique_keys(financials_df, ["Customer_ID", "snapshot_date"], "Silver Financials")
    attributes_count = attributes_df.count()
    financials_count = financials_df.count()

    # 3. Join the application base
    application_base_df = (
        attributes_df
        .join(
            financials_df,
            on=[
                "Customer_ID",
                "snapshot_date",
            ],
            how="inner",
        )
        .withColumnRenamed(
            "snapshot_date",
            "application_date",
        )
    )
    if application_base_df.count() != attributes_count or attributes_count != financials_count:
        raise ValueError(
            f"Gold features {application_date_str}: Attributes/Financials unmatched "
            f"or fan-out ({attributes_count} / {financials_count})"
        )

    # 4. Create Attributes and Financials Gold features
    gold_base_df = _add_financial_features(
        application_base_df
    )

    # 5. Identify available prior-three-month Clickstream partitions
    prior_month_dates = [
        (
            application_date
            - relativedelta(months=month_offset)
        ).strftime("%Y-%m-%d")
        for month_offset in range(3, 0, -1)
    ]

    clickstream_paths = []

    for snapshot_date_str in prior_month_dates:
        snapshot_suffix = snapshot_date_str.replace(
            "-",
            "_",
        )
        clickstream_path = os.path.join(
            silver_clickstream_directory,
            (
                "silver_clickstream_"
                f"{snapshot_suffix}.parquet"
            ),
        )

        if os.path.exists(clickstream_path):
            clickstream_paths.append(clickstream_path)

    # 6. Aggregate available Clickstream history
    clickstream_aggregated_df = _aggregate_clickstream(
        gold_base_df,
        clickstream_paths,
        spark,
    )

    # 7. Left join Clickstream and handle missing history
    if clickstream_aggregated_df is None:
        gold_df = _add_empty_clickstream_features(
            gold_base_df
        )
    else:
        gold_df = (
            gold_base_df
            .join(
                clickstream_aggregated_df,
                on=[
                    "Customer_ID",
                    "application_date",
                ],
                how="left",
            )
            .withColumn(
                "clickstream_months_available",
                F.coalesce(
                    F.col(
                        "clickstream_months_available"
                    ),
                    F.lit(0),
                ).cast("int"),
            )
            .withColumn(
                "clickstream_history_available",
                F.coalesce(
                    F.col(
                        "clickstream_history_available"
                    ),
                    F.lit(0),
                ).cast("int"),
            )
        )

    # 8. Enforce the ordered Gold output contract
    gold_df = gold_df.select(*FINAL_GOLD_COLUMNS)
    unique_keys(gold_df, ["Customer_ID", "application_date"], "Gold features")
    if gold_df.count() != attributes_count:
        raise ValueError(f"Gold features {application_date_str}: join lost or duplicated rows")

    if len(gold_df.columns) != 106:
        raise ValueError(
            "Gold Feature Store must contain 106 columns."
        )

    # 9. Save Gold
    os.makedirs(
        gold_feature_store_directory,
        exist_ok=True,
    )
    gold_partition_name = (
        f"gold_feature_store_{date_suffix}.parquet"
    )
    gold_filepath = os.path.join(
        gold_feature_store_directory,
        gold_partition_name,
    )

    write_spark_parquet_local(gold_df, Path(gold_filepath))
    print(
        "saved to:",
        gold_filepath,
    )

    return gold_df
