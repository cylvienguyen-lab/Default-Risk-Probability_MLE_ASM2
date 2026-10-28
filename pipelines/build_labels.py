"""Build and publish Gold Labels by explicit outcome snapshot month."""

import argparse
import shutil
import uuid
from pathlib import Path

from pipelines.utils import data_processing_bronze_lms as bronze
from pipelines.utils import data_processing_gold_label as gold
from pipelines.utils import data_processing_silver_lms as silver
from pipelines.utils.artifacts import previous_publication, publish_partition, run_record, sha256
from pipelines.utils.config import load_config, location, month, months, partition
from pipelines.utils.validation import label_available_date, validate_label_partition


def run_snapshot(snapshot_month: str, config: dict | str | Path,
                 run_id: str, spark=None) -> dict:
    from pyspark.sql import SparkSession

    cfg = load_config(config) if not isinstance(config, dict) else config
    snapshot = month(snapshot_month)
    owned = spark is None
    if owned:
        spark = (SparkSession.builder.appName("build_labels")
                 .master(cfg["project"]["runtime"]["spark_master"])
                 .config("spark.sql.shuffle.partitions",
                         cfg["project"]["runtime"]["spark_shuffle_partitions"]).getOrCreate())
    staging_root = cfg["root"] / ".staging" / f"labels_{run_id}_{uuid.uuid4().hex}"
    try:
        # 1. Ingest and validate the selected raw LMS snapshot
        source = location(cfg, "lms_source")
        bronze.process_bronze_table(snapshot_month, source, location(cfg, "bronze_lms"),
                                    spark, cfg["project"]["policy"]["max_bronze_rows"])

        # 2. Compute valid monthly DPD and MOB-6 labels
        silver.process_silver_table(snapshot_month, location(cfg, "bronze_lms"),
                                    location(cfg, "silver_lms"), spark)
        target = cfg["project"]["target"]
        frame = gold.process_labels_gold_table(
            snapshot_month, location(cfg, "silver_lms"), staging_root, spark,
            target["dpd_threshold"], target["horizon_months"])
        audit = validate_label_partition(frame, snapshot_month, target["label_def"])

        # 3. Publish this outcome snapshot and its availability metadata
        staged = partition(staging_root, "gold_label_store", snapshot_month, "parquet")
        output = partition(location(cfg, "gold_labels"), "gold_label_store",
                           snapshot_month, "parquet")
        publish_partition(staged, output)
        available = label_available_date(snapshot, target["reporting_lag_months"])
        source_ref = {"lms": {"path": str(source), "sha256": sha256(source)}}
        revision = previous_publication(cfg, "build_labels", snapshot_month, source_ref)
        affected = (["training eligibility and frozen Test/OOT evaluation"]
                    if snapshot < month("2025-01-01") else
                    ["mature-label monitoring for linked application cohorts"])
        manifest = run_record(cfg, run_id, "build_labels", snapshot_month, "published",
                              outcome_snapshot_month=snapshot_month,
                              label_available_date=available.isoformat(),
                              published_paths=[str(output)], output_counts=audit,
                              input_refs=source_ref, revision=revision,
                              affected_downstream=affected,
                              empty_reason="no MOB-6 loans" if audit["rows"] == 0 else None)
        return {"run_id": run_id, "job_name": "build_labels", "status": "published",
                "outcome_snapshot_month": snapshot_month, "published_paths": [str(output)],
                "output_counts": audit, "label_available_date": available.isoformat(),
                "input_refs": source_ref, "revision": revision,
                "affected_downstream": affected, "manifest_path": str(manifest),
                "audit_path": str(manifest)}
    except Exception as error:
        run_record(cfg, run_id, "build_labels", snapshot_month, "failed",
                   outcome_snapshot_month=snapshot_month, error=str(error))
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
        spark = (SparkSession.builder.appName("build_labels_range")
                 .master(cfg["project"]["runtime"]["spark_master"])
                 .config("spark.sql.shuffle.partitions",
                         cfg["project"]["runtime"]["spark_shuffle_partitions"]).getOrCreate())
    try:
        return [run_snapshot(value, cfg, run_id, spark)
                for value in months(start, end_exclusive)]
    finally:
        if owned:
            spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--outcome-snapshot-month")
    scope.add_argument("--start")
    parser.add_argument("--end-exclusive")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = args.config or load_config()["config_path"]
    if args.outcome_snapshot_month:
        if args.end_exclusive:
            parser.error("--end-exclusive requires --start")
        print(run_snapshot(args.outcome_snapshot_month, config, args.run_id))
    else:
        if not args.end_exclusive:
            parser.error("--start requires --end-exclusive")
        print(run_range(args.start, args.end_exclusive, config, args.run_id))


if __name__ == "__main__":
    main()
