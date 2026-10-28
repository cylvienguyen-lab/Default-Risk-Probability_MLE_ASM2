"""Score application cohorts with one frozen model and publish monthly partitions."""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipelines.utils.artifacts import publish_partition, read_released_model, run_record, sha256, sha256_tree
from pipelines.utils.config import load_config, location, month, months, partition
from pipelines.utils.model_input import prepare_model_input
from pipelines.utils.validation import validate_model_bundle, validate_predictions


def resolve_model(config: dict, model_version: str | None = None,
                  model_path: Path | None = None) -> tuple[dict, Path]:
    # 1. Resolve the released pointer once or honor an explicit immutable reference
    if model_path:
        path = Path(model_path).resolve()
    elif model_version:
        path = location(config, "model_bank") / model_version / "model.joblib"
    else:
        path = Path(read_released_model(location(config, "model_bank"))["artifact_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact absent: {path}")
    bundle = joblib.load(path)
    validate_model_bundle(bundle)
    if model_version and bundle["model_version"] != model_version:
        raise ValueError("Requested model version and artifact disagree")
    return bundle, path


def score_month(application_month: str, config: dict | str | Path,
                run_id: str, bundle: dict, model_ref: Path) -> dict:
    cfg = load_config(config) if not isinstance(config, dict) else config
    month(application_month)
    validate_model_bundle(bundle)
    version = bundle["model_version"]

    # 1. Read and validate this application cohort only
    feature_path = partition(location(cfg, "gold_features"), "gold_feature_store",
                             application_month, "parquet")
    if not feature_path.exists():
        raise FileNotFoundError(f"Inference feature partition absent: {feature_path}")
    feature_hash = sha256_tree(feature_path)
    feature_ref = str(feature_path.relative_to(cfg["root"]))
    features = pd.read_parquet(feature_path)
    if list(features.columns) != cfg["feature"]["gold_columns"]:
        raise ValueError("Inference: Gold feature schema mismatch")
    if features.empty or features[["Customer_ID", "application_date"]].isna().any().any():
        raise ValueError(f"Inference {application_month}: empty cohort or null keys")
    if features.duplicated(["Customer_ID", "application_date"]).any():
        raise ValueError(f"Inference {application_month}: duplicate application key")
    if not pd.to_datetime(features["application_date"]).dt.date.eq(month(application_month)).all():
        raise ValueError(f"Inference {application_month}: feature partition contains other cohorts")

    # 2. Prepare the bundle-frozen model input and score positive class 1
    model_input = prepare_model_input(features, bundle)
    probabilities = bundle["model"].predict_proba(model_input)[:, 1]
    predictions = features[["Customer_ID", "application_date"]].copy()
    predictions["model_version"] = version
    predictions["p_default"] = probabilities
    predictions["prediction_as_of_date"] = month(application_month)
    predictions["scored_at"] = datetime.now(timezone.utc).isoformat()
    predictions["inference_run_id"] = run_id
    predictions["feature_input_ref"] = f"{feature_ref}#sha256={feature_hash}"
    audit = validate_predictions(predictions, features, version)

    # 3. Persist a staging reference outside canonical read paths
    staging = (cfg["root"] / ".staging" / "inference" / run_id / version /
               application_month[:7] / "predictions.parquet")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise FileExistsError(f"Inference staging already exists: {staging}")
    predictions.to_parquet(staging, index=False)
    return {"run_id": run_id, "job_name": "score_month", "status": "staged",
            "application_month": application_month, "model_version": version,
            "staging_paths": [str(staging)], "input_refs": {
                "features": feature_ref, "feature_sha256": feature_hash,
                "model": str(model_ref.relative_to(cfg["root"]))
                if model_ref.is_relative_to(cfg["root"]) else str(model_ref),
                "model_sha256": sha256(model_ref)}, "output_counts": audit}


def publish_month(staging_ref: dict, config: dict | str | Path) -> dict:
    cfg = load_config(config) if not isinstance(config, dict) else config
    application_month = staging_ref["application_month"]
    version = staging_ref["model_version"]
    staging = Path(staging_ref["staging_paths"][0])
    feature_path = Path(staging_ref["input_refs"]["features"])
    if not feature_path.is_absolute():
        feature_path = cfg["root"] / feature_path

    # 1. Validate the durable staging output against the exact input cohort
    predictions = pd.read_parquet(staging)
    features = pd.read_parquet(feature_path, columns=["Customer_ID", "application_date"])
    if sha256_tree(feature_path) != staging_ref["input_refs"]["feature_sha256"]:
        raise ValueError("Inference: feature partition changed after scoring")
    audit = validate_predictions(predictions, features, version)
    if len(predictions) != staging_ref["output_counts"]["rows"]:
        raise ValueError("Inference: staging row count changed before publication")

    # 2. Publish only the requested model/cohort partition
    target = (location(cfg, "predictions") / version / application_month[:7] /
              "predictions.parquet")
    publish_partition(staging, target)
    manifest = run_record(cfg, staging_ref["run_id"], "inference", application_month,
                          "published", application_month=application_month,
                          model_version=version, published_paths=[str(target)],
                          input_refs=staging_ref["input_refs"], output_counts=audit,
                          prediction_as_of_date=application_month,
                          scored_at=predictions["scored_at"].iloc[0])
    return {"run_id": staging_ref["run_id"], "job_name": "inference",
            "status": "published", "application_month": application_month,
            "model_version": version, "published_paths": [str(target)],
            "input_refs": staging_ref["input_refs"], "output_counts": audit,
            "manifest_path": str(manifest), "audit_path": str(manifest)}


def run(config: dict | str | Path, run_id: str, application_month: str | None = None,
        start: str | None = None, end_exclusive: str | None = None,
        model_version: str | None = None, model_path: Path | None = None) -> list[dict]:
    cfg = load_config(config) if not isinstance(config, dict) else config
    if application_month and (start or end_exclusive):
        raise ValueError("Select one application month or one half-open range")
    if application_month:
        requested = [month(application_month).isoformat()]
    else:
        if (start is None) != (end_exclusive is None):
            raise ValueError("Both start and end-exclusive are required")
        defaults = cfg["project"]["splits"]["inference"]
        requested = months(start or defaults[0], end_exclusive or defaults[1])
    # 1. Freeze the model reference for the entire run
    bundle, path = resolve_model(cfg, model_version, model_path)
    # 2. Score and publish each requested month independently
    results = []
    for value in requested:
        try:
            staged = score_month(value, cfg, run_id, bundle, path)
            results.append(publish_month(staged, cfg))
        except Exception as error:
            run_record(cfg, run_id, "inference", value, "failed",
                       application_month=value, model_version=bundle["model_version"],
                       input_refs={"model": str(path)}, error=str(error))
            raise
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--application-month")
    scope.add_argument("--start")
    parser.add_argument("--end-exclusive")
    parser.add_argument("--model-version")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.start and not args.end_exclusive:
        parser.error("--start requires --end-exclusive")
    if args.application_month and args.end_exclusive:
        parser.error("--end-exclusive requires --start")
    print(json.dumps(run(args.config or load_config()["config_path"], args.run_id,
                         args.application_month, args.start, args.end_exclusive,
                         args.model_version, args.model_path), indent=2))


if __name__ == "__main__":
    main()
