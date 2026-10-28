"""Refit the frozen CatBoost on eligible Development applications."""

import argparse
import json
from importlib.metadata import version as package_version
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)

from pipelines.utils.artifacts import release_model, run_record, sha256, sha256_tree
from pipelines.utils.config import add_months, load_config, location, months, partition
from pipelines.utils.model_input import prepare_model_input, selected_features
from pipelines.utils.validation import (validate_fit_eligibility, validate_model_bundle,
                              validate_training_stores)


def assemble_datasets(config: dict) -> tuple[pd.DataFrame, dict]:
    """Read only the frozen Development/Test/OOT populations and their labels."""
    project = config["project"]
    scopes = ("development", "test", "oot")
    feature_months = [value for scope in scopes
                      for value in months(*project["splits"][scope])]
    label_months = sorted({add_months(pd.Timestamp(value).date(),
                        project["target"]["horizon_months"]).isoformat()
                        for value in feature_months})
    feature_paths = [partition(location(config, "gold_features"), "gold_feature_store",
                               value, "parquet") for value in feature_months]
    label_paths = [partition(location(config, "gold_labels"), "gold_label_store",
                             value, "parquet") for value in label_months]
    for path in feature_paths + label_paths:
        if not path.exists():
            raise FileNotFoundError(f"Training input partition absent: {path}")
    features = pd.concat([pd.read_parquet(path) for path in feature_paths], ignore_index=True)
    labels = pd.concat([pd.read_parquet(path) for path in label_paths], ignore_index=True)
    if list(features.columns) != config["feature"]["gold_columns"]:
        raise ValueError("Training Feature Store does not match ordered Gold contract")
    assembled = validate_training_stores(features, labels, project)
    refs = {"features": [{"path": str(path), "sha256": sha256_tree(path)}
                          for path in feature_paths],
            "labels": [{"path": str(path), "sha256": sha256_tree(path)}
                        for path in label_paths]}
    return assembled, refs


def fit_and_evaluate(assembled: pd.DataFrame, config: dict, fit_as_of: str) -> tuple[dict, pd.DataFrame]:
    # 1. Select only eligible Development rows and frozen model inputs
    development = validate_fit_eligibility(assembled, fit_as_of)
    spec = selected_features(config["feature"], config["model"])
    features = prepare_model_input(development, spec)
    target = development["label"].astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("Final fit requires both Development target classes")

    # 2. Fit the agreed single CatBoost model
    params = config["model"]["parameters"]
    estimator = CatBoostClassifier(**params, verbose=False,
                                   allow_writing_files=False,
                                   cat_features=spec["categorical_columns"])
    estimator.fit(features, target)

    # 3. Evaluate the frozen Test/OOT cohorts retrospectively
    evaluation = {}
    row_outputs = []
    for scope in ("test", "oot"):
        subset = assembled[assembled["split"].eq(scope)]
        if subset.empty or subset["label"].isna().any():
            raise ValueError(f"Evaluation: missing required {scope} rows or labels")
        input_frame = prepare_model_input(subset, spec)
        y_true = subset["label"].astype(int)
        probabilities = estimator.predict_proba(input_frame)[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Evaluation {scope}: non-finite probabilities")
        metrics = {"Log Loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
                   "Brier Score": float(brier_score_loss(y_true, probabilities)),
                   "ROC-AUC": float(roc_auc_score(y_true, probabilities))
                   if y_true.nunique() == 2 else None,
                   "Average Precision": float(average_precision_score(y_true, probabilities))
                   if y_true.nunique() == 2 else None,
                   "rows": int(len(subset)), "defaults": int(y_true.sum())}
        evaluation[scope] = metrics
        scored = subset[["Customer_ID", "application_date", "label"]].copy()
        scored["split"] = scope
        scored["p_default"] = probabilities
        row_outputs.append(scored)
    bundle = {
        "model_version": config["model"]["model_version"], "model_name": "CatBoost",
        "model": estimator, **spec, "positive_class": 1,
        "target_column": "label", "label_def": config["project"]["target"]["label_def"],
        "preprocessing": {"numeric_values": config["model"]["numeric_values"],
                          "categorical_values": config["model"]["categorical_values"]},
        "hp_params": params, "schema_version": config["feature"]["schema_version"],
        "simulated_fit_as_of": fit_as_of,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_evidence": config["model"]["selection_evidence"],
        "data_dates": {"development_start": config["project"]["splits"]["development"][0],
                       "development_end_exclusive": config["project"]["splits"]["development"][1]},
        "data_stats": {"development_rows": int(len(development)),
                       "development_defaults": int(target.sum())},
        "results": evaluation,
    }
    validate_model_bundle(bundle)
    return bundle, pd.concat(row_outputs, ignore_index=True)


def save_candidate(bundle: dict, evaluation_rows: pd.DataFrame,
                   assembled: pd.DataFrame,
                   config: dict, run_id: str, input_refs: dict) -> dict:
    # 1. Persist candidate without touching the release pointer
    version = bundle["model_version"]
    bundle["input_refs"] = input_refs
    bundle["environment"] = {name: package_version(name) for name in
                             ("catboost", "numpy", "pandas", "pyarrow", "scikit-learn", "joblib")}
    candidate = location(config, "model_bank") / "candidates" / version / f"{run_id}.joblib"
    if candidate.exists():
        raise FileExistsError(f"Candidate already exists: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, candidate)
    loaded = joblib.load(candidate)
    validate_model_bundle(loaded)
    sample = prepare_model_input(assembled[assembled["split"].eq("development")].head(10), bundle)
    np.testing.assert_allclose(loaded["model"].predict_proba(sample)[:, 1],
                               bundle["model"].predict_proba(sample)[:, 1],
                               rtol=0, atol=1e-12)
    report_dir = location(config, "evaluation") / version
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = report_dir / f"{run_id}_rows.parquet"
    evaluation_rows.to_parquet(evaluation_path, index=False)
    membership = assembled[["Customer_ID", "application_date", "split",
                            "label_available_date", "label"]].copy()
    membership["included_in_final_fit"] = membership["split"].eq("development") & (
        membership["label_available_date"] <= pd.Timestamp(
            bundle["simulated_fit_as_of"]).date())
    membership_path = report_dir / f"{run_id}_membership.parquet"
    membership.to_parquet(membership_path, index=False)
    manifest = run_record(config, run_id, "train", version, "candidate",
                          model_version=version, staging_paths=[str(candidate)],
                          input_refs=input_refs, output_counts=bundle["data_stats"],
                          evaluation=bundle["results"], evaluation_ref=str(evaluation_path),
                          membership_ref=str(membership_path),
                          candidate_sha256=sha256(candidate))
    return {"run_id": run_id, "job_name": "train", "status": "candidate",
            "model_version": version, "staging_paths": [str(candidate)],
            "input_refs": input_refs, "output_counts": bundle["data_stats"],
            "manifest_path": str(manifest), "audit_path": str(manifest),
            "evaluation_ref": str(evaluation_path),
            "membership_ref": str(membership_path)}


def publish_candidate(candidate: Path, config: dict | str | Path, run_id: str,
                      candidate_manifest: str | None = None) -> dict:
    cfg = load_config(config) if not isinstance(config, dict) else config
    bundle = joblib.load(candidate)
    validate_model_bundle(bundle)
    pointer = release_model(candidate, bundle["model_version"],
                            location(cfg, "model_bank"),
                            {"run_id": run_id, "status": "released",
                             "source_candidate": str(candidate),
                              "candidate_manifest": candidate_manifest,
                             "simulated_fit_as_of": bundle["simulated_fit_as_of"],
                             "trained_at": bundle["trained_at"]})
    return {"run_id": run_id, "job_name": "publish_model", "status": "released",
            "model_version": bundle["model_version"], "published_paths": [pointer["artifact_path"]],
            "manifest_path": pointer["manifest_path"], "audit_path": pointer["manifest_path"]}


def run(config: dict | str | Path, run_id: str, fit_as_of: str | None = None,
        publish: bool = False) -> dict:
    cfg = load_config(config) if not isinstance(config, dict) else config
    cutoff = fit_as_of or cfg["project"]["final_fit_as_of"]
    try:
        # 1. Assemble validated Development/Test/OOT membership
        assembled, refs = assemble_datasets(cfg)
        # 2. Fit and evaluate the frozen specification
        bundle, rows = fit_and_evaluate(assembled, cfg, cutoff)
        # 3. Save candidate and optionally release after validation
        result = save_candidate(bundle, rows, assembled, cfg, run_id, refs)
        if publish:
            result["release"] = publish_candidate(Path(result["staging_paths"][0]), cfg,
                                                   run_id, result["manifest_path"])
        return result
    except Exception as error:
        run_record(cfg, run_id, "train", cfg["model"]["model_version"], "failed",
                   model_version=cfg["model"]["model_version"],
                   simulated_fit_as_of=cutoff, error=str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-as-of")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config or load_config()["config_path"], args.run_id,
                         args.fit_as_of, args.publish), indent=2))


if __name__ == "__main__":
    main()
