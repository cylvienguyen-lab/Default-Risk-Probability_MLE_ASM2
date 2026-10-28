"""Build and publish one or more monthly Gold Feature partitions."""

import argparse
import json
import shutil
import uuid
from pathlib import Path

from pipelines.utils import data_processing_bronze_features as bronze
from pipelines.utils import data_processing_gold_features as gold
from pipelines.utils import data_processing_silver_features as silver
from pipelines.utils.artifacts import (previous_publication, publish_partition, run_record,
                             sha256, write_manifest)
from pipelines.utils.config import add_months, load_config, location, month, months, partition
from pipelines.utils.validation import audit_invalid_to_null, validate_feature_partition


def run_month(application_month: str, config: dict | str | Path,
              run_id: str, spark=None) -> dict:
    from pyspark.sql import SparkSession

    cfg = load_config(config) if not isinstance(config, dict) else config
    application_date = month(application_month)
    owned = spark is None
    if owned:
        spark = (SparkSession.builder.appName("build_features")
                 .master(cfg["project"]["runtime"]["spark_master"])
                 .config("spark.sql.shuffle.partitions",
                         cfg["project"]["runtime"]["spark_shuffle_partitions"]).getOrCreate())
    staging_root = cfg["root"] / ".staging" / f"features_{run_id}_{uuid.uuid4().hex}"
    try:
        # 1. Ingest and clean the application snapshot
        sources = {"attributes": "attributes_source", "financials": "financials_source"}
        bronze_frames = {}
        for name, source_key in sources.items():
            bronze_frames[name] = bronze.process_bronze_feature(
                application_month, location(cfg, source_key), location(cfg, f"bronze_{name}"),
                name, spark, cfg["project"]["policy"]["max_bronze_rows"])
        attributes = silver.process_silver_attributes(
            application_month, location(cfg, "bronze_attributes"),
            location(cfg, "silver_attributes"), spark)
        financials = silver.process_silver_financials(
            application_month, location(cfg, "bronze_financials"),
            location(cfg, "silver_financials"), spark)
        null_audit = {
            "attributes": audit_invalid_to_null(
                bronze_frames["attributes"], attributes, {"Age": "Age"},
                "Silver Attributes"),
            "financials": audit_invalid_to_null(
                bronze_frames["financials"], financials,
                {name: name for name in (
                    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
                    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                    "Delay_from_due_date", "Num_of_Delayed_Payment",
                    "Changed_Credit_Limit", "Num_Credit_Inquiries",
                    "Outstanding_Debt", "Credit_Utilization_Ratio",
                    "Total_EMI_per_month", "Amount_invested_monthly",
                    "Monthly_Balance")}, "Silver Financials")}

        # 2. Build the prior-three-month Clickstream dependencies
        coverage_start = month(cfg["project"]["source_coverage"]["clickstream"][0])
        clickstream_source = location(cfg, "clickstream_source")
        clickstream_hash = sha256(clickstream_source)
        for offset in (3, 2, 1):
            history_month = add_months(application_date, -offset)
            if history_month < coverage_start:
                continue
            history = history_month.isoformat()
            silver_path = partition(location(cfg, "silver_clickstream"),
                                    "silver_clickstream", history, "parquet")
            source_record = silver_path.with_suffix(".source.json")
            cached = (silver_path.exists() and source_record.exists() and
                      json.loads(source_record.read_text(encoding="utf-8"))
                      .get("source_sha256") == clickstream_hash)
            if not cached:
                bronze.process_bronze_feature(
                    history, clickstream_source, location(cfg, "bronze_clickstream"),
                    "clickstream", spark, cfg["project"]["policy"]["max_bronze_rows"])
                silver.process_silver_clickstream(history, location(cfg, "bronze_clickstream"),
                                                  location(cfg, "silver_clickstream"), spark)
                write_manifest(source_record, {"snapshot_month": history,
                                               "source_path": str(clickstream_source),
                                               "source_sha256": clickstream_hash})

        # 3. Build, validate, and publish only this Gold month
        frame = gold.process_gold_feature_store(
            application_month, location(cfg, "silver_attributes"),
            location(cfg, "silver_financials"), location(cfg, "silver_clickstream"),
            staging_root, spark)
        audit = validate_feature_partition(frame, cfg["feature"], application_month)
        source_columns = cfg["feature"]["gold_columns"]
        if gold.FINAL_GOLD_COLUMNS != source_columns:
            raise ValueError("Gold producer and feature config disagree on ordered schema")
        staged = partition(staging_root, "gold_feature_store", application_month, "parquet")
        target = partition(location(cfg, "gold_features"), "gold_feature_store",
                           application_month, "parquet")
        publish_partition(staged, target)
        source_refs = {name: {"path": str(location(cfg, source_key)),
                              "sha256": sha256(location(cfg, source_key))}
                       for name, source_key in {**sources,
                       "clickstream": "clickstream_source"}.items()}
        revision = previous_publication(cfg, "build_features", application_month,
                                        source_refs)
        affected = (["training candidate and frozen Test/OOT evaluation"]
                    if application_date < month(cfg["project"]["splits"]["inference"][0])
                    else [f"predictions for application month {application_month}",
                          "monitoring derived from those predictions"])
        manifest = run_record(cfg, run_id, "build_features", application_month, "published",
                              application_month=application_month, published_paths=[str(target)],
                              input_refs=source_refs, output_counts=audit,
                              silver_invalid_to_null_counts=null_audit,
                              revision=revision, affected_downstream=affected)
        return {"run_id": run_id, "job_name": "build_features", "status": "published",
                "application_month": application_month, "published_paths": [str(target)],
                "input_refs": source_refs, "output_counts": audit,
                "silver_invalid_to_null_counts": null_audit,
                "revision": revision, "affected_downstream": affected,
                "manifest_path": str(manifest), "audit_path": str(manifest)}
    except Exception as error:
        run_record(cfg, run_id, "build_features", application_month, "failed",
                   application_month=application_month, error=str(error))
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if owned:
            spark.stop()


def run_range(start: str, end_exclusive: str, config: dict | str | Path,
              run_id: str, spark=None) -> list[dict]:
    from pyspark.sql import SparkSession

    cfg = load_config(config) if not isinstance(config, dict) else config
    owned = spark is None
    if owned:
        spark = (SparkSession.builder.appName("build_features_range")
                 .master(cfg["project"]["runtime"]["spark_master"])
                 .config("spark.sql.shuffle.partitions",
                         cfg["project"]["runtime"]["spark_shuffle_partitions"]).getOrCreate())
    try:
        return [run_month(value, cfg, run_id, spark) for value in months(start, end_exclusive)]
    finally:
        if owned:
            spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--application-month")
    scope.add_argument("--start")
    parser.add_argument("--end-exclusive")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = args.config or load_config()["config_path"]
    if args.application_month:
        if args.end_exclusive:
            parser.error("--end-exclusive requires --start")
        print(run_month(args.application_month, config, args.run_id))
    else:
        if not args.end_exclusive:
            parser.error("--start requires --end-exclusive")
        print(run_range(args.start, args.end_exclusive, config, args.run_id))


if __name__ == "__main__":
    main()
