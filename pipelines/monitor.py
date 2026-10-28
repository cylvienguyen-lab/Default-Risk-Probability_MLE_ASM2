"""Monitor published monthly predictions against a frozen reference and mature labels."""

import argparse
import json
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)

from pipelines.inference import resolve_model
from pipelines.utils.artifacts import (publish_partition, run_record, sha256,
                                       sha256_tree)
from pipelines.utils.config import add_months, load_config, location, month, months, partition
from pipelines.utils.model_input import prepare_model_input
from pipelines.utils.validation import (label_available_date, require_columns,
                                        validate_predictions)


PREDICTION_COLUMNS = ["Customer_ID", "application_date", "model_version",
                      "p_default", "prediction_as_of_date", "scored_at",
                      "inference_run_id", "feature_input_ref"]
LABEL_COLUMNS = ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]
KEYS = ["Customer_ID", "application_date"]
METRICS = ["log_loss", "brier_score", "roc_auc", "average_precision",
           "actual_default_rate", "mean_predicted_probability"]
OUTPUT_COLUMNS = ["application_month", "model_version", "monitoring_as_of",
                  "reference_version", "signal_group", "signal_name", "value",
                  "status", "reason", "n_predictions", "n_expected_eligible",
                  "n_matched_labels", "label_coverage", "n_reference", "n_current",
                  "run_id", "feature_ref", "prediction_ref", "label_ref",
                  "model_ref", "reference_ref", "monitoring_config_ref"]


def load_monitoring_config(config: dict, path: Path | str | None = None) -> tuple[dict, Path]:
    """Load the separate monitoring policy from the project config directory."""
    source = Path(path).resolve() if path else (
        config["config_path"].parent / config["project"]["monitoring_config"])
    policy = json.loads(source.read_text(encoding="utf-8"))
    _reference_version(policy["reference_version"])
    if policy["numeric_quantile_bins"] < 2 or policy["calibration_bins"] < 2:
        raise ValueError("Monitoring bins must be at least two")
    if policy["psi_smoothing_count"] <= 0 or policy["minimum_stability_sample_size"] < 1:
        raise ValueError("Monitoring smoothing/sample policy invalid")
    return policy, source


def _reference_version(value: str) -> str:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Monitoring reference version must be path-safe")
    return value


def reference_path(config: dict, model_version: str, reference_version: str) -> Path:
    return (location(config, "monitoring_reports") / "references" / model_version /
            f"{reference_version}.json")


def monitoring_path(config: dict, model_version: str, reference_version: str,
                    monitoring_as_of: str, application_month: str) -> Path:
    return (location(config, "monitoring") / model_version / reference_version /
            monitoring_as_of / application_month[:7] / "monitoring.parquet")


def _remove_empty_staging_parents(path: Path, config: dict) -> None:
    staging_root = config["root"] / ".staging"
    if not path.is_relative_to(staging_root):
        raise ValueError(f"Monitoring staging path outside project staging: {path}")
    for parent in path.parents:
        if parent == staging_root:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _read_feature_month(config: dict, application_month: str, bundle: dict) -> tuple[pd.DataFrame, dict]:
    path = partition(location(config, "gold_features"), "gold_feature_store",
                     application_month, "parquet")
    if not path.exists():
        raise FileNotFoundError(f"Monitoring feature partition absent: {path}")
    frame = pd.read_parquet(path)
    require_columns(frame, config["feature"]["gold_columns"],
                    "Monitoring Gold Features", exact=True)
    if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"Monitoring {application_month}: empty/null/duplicate feature keys")
    if frame["Customer_ID"].duplicated().any():
        raise ValueError("Monitoring: multiple applications per customer unsupported")
    dates = pd.to_datetime(frame["application_date"], errors="raise").dt.date
    if not dates.eq(month(application_month)).all():
        raise ValueError(f"Monitoring {application_month}: wrong feature cohort")
    prepare_model_input(frame, bundle)
    return frame, {"path": str(path), "sha256": sha256_tree(path)}


def _read_feature_range(config: dict, start: str, end_exclusive: str,
                        bundle: dict) -> tuple[pd.DataFrame, list[dict]]:
    batches = [_read_feature_month(config, value, bundle)
               for value in months(start, end_exclusive)]
    frames = pd.concat([frame for frame, _ in batches], ignore_index=True)
    if frames["Customer_ID"].duplicated().any():
        raise ValueError("Monitoring reference: multiple applications per customer")
    return frames, [ref for _, ref in batches]


def _numeric_profile(values: pd.Series, bins: int) -> dict:
    numbers = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numbers).any():
        raise ValueError("Monitoring reference: infinite numeric input")
    valid = numbers[~np.isnan(numbers)]
    if len(valid) == 0:
        edges = []
    else:
        quantiles = np.quantile(valid, np.linspace(0, 1, bins + 1)[1:-1])
        edges = np.unique(quantiles).astype(float).tolist()
    indexes = np.searchsorted(edges, valid, side="right")
    counts = {f"bin_{index}": int((indexes == index).sum())
              for index in range(len(edges) + 1)}
    counts["missing"] = int(np.isnan(numbers).sum())
    return {"type": "numeric", "edges": edges, "counts": counts,
            "n_reference": int(len(numbers))}


def _categorical_profile(values: pd.Series) -> dict:
    present = values.dropna().astype(str)
    categories = sorted(present.unique().tolist())
    counts = {f"value:{category}": int(present.eq(category).sum())
              for category in categories}
    counts.update({"unseen": 0, "missing": int(values.isna().sum())})
    return {"type": "categorical", "categories": categories,
            "counts": counts, "n_reference": int(len(values))}


def _current_counts(values: pd.Series, profile: dict) -> dict:
    counts = dict.fromkeys(profile["counts"], 0)
    if profile["type"] == "numeric":
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numbers).any():
            raise ValueError("Monitoring: infinite numeric input")
        valid = numbers[~np.isnan(numbers)]
        indexes = np.searchsorted(profile["edges"], valid, side="right")
        for index in range(len(profile["edges"]) + 1):
            counts[f"bin_{index}"] = int((indexes == index).sum())
        counts["missing"] = int(np.isnan(numbers).sum())
    else:
        counts["missing"] = int(values.isna().sum())
        for value, count in values.dropna().astype(str).value_counts().items():
            key = f"value:{value}"
            counts[key if key in counts else "unseen"] += int(count)
    if sum(counts.values()) != len(values):
        raise ValueError("Monitoring: distribution row reconciliation failed")
    return counts


def population_stability_index(values: pd.Series, profile: dict, smoothing: float) -> float:
    """Compare fixed reference bins with additive-count smoothing."""
    current = _current_counts(values, profile)
    keys = list(profile["counts"])
    n_reference = profile["n_reference"]
    n_current = len(values)
    k = len(keys)
    if n_reference == 0 or n_current == 0:
        raise ValueError("PSI requires nonempty reference and current populations")
    expected = np.array([(profile["counts"][key] + smoothing) /
                         (n_reference + smoothing * k) for key in keys])
    actual = np.array([(current[key] + smoothing) /
                       (n_current + smoothing * k) for key in keys])
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_reference(config: dict | str | Path, run_id: str,
                    model_version: str | None = None, model_path: Path | None = None,
                    monitoring_config: Path | str | None = None,
                    reference_version: str | None = None) -> dict:
    """Freeze feature and score distributions without fitting a model."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    policy, policy_path = load_monitoring_config(cfg, monitoring_config)
    if reference_version:
        policy = {**policy, "reference_version": _reference_version(reference_version)}
    bundle, model_ref = resolve_model(cfg, model_version, model_path)
    version = bundle["model_version"]
    target = reference_path(cfg, version, policy["reference_version"])
    if target.exists():
        raise FileExistsError(f"Monitoring reference is immutable: {target}")

    # 1. Read fixed Development features and fixed OOT score population
    feature_scope = policy["feature_reference"]
    score_scope = policy["score_reference"]
    development, feature_refs = _read_feature_range(
        cfg, feature_scope["start"], feature_scope["end_exclusive"], bundle)
    oot, score_refs = _read_feature_range(
        cfg, score_scope["start"], score_scope["end_exclusive"], bundle)

    # 2. Freeze selected-feature bins and score bins using the released model
    profiles = {name: _numeric_profile(development[name], policy["numeric_quantile_bins"])
                for name in bundle["numeric_columns"]}
    profiles.update({name: _categorical_profile(development[name])
                     for name in bundle["categorical_columns"]})
    scores = bundle["model"].predict_proba(prepare_model_input(oot, bundle))[:, 1]
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Monitoring reference: invalid frozen model scores")
    profiles["p_default"] = _numeric_profile(pd.Series(scores), policy["numeric_quantile_bins"])
    reference = {
        "reference_version": policy["reference_version"], "model_version": version,
        "model_ref": str(model_ref), "model_sha256": sha256(model_ref),
        "monitoring_config_ref": str(policy_path),
        "monitoring_config_sha256": sha256(policy_path),
        "feature_columns": bundle["feature_columns"],
        "numeric_columns": bundle["numeric_columns"],
        "categorical_columns": bundle["categorical_columns"],
        "feature_reference": {**feature_scope, "rows": len(development), "input_refs": feature_refs},
        "score_reference": {**score_scope, "rows": len(oot), "input_refs": score_refs,
                            "scoring": "saved frozen model; no fit or calibration"},
        "method": {"name": "PSI", "numeric_bins": "fixed Development quantile edges",
                   "category_bins": "Development categories plus unseen/missing",
                   "smoothing_count": policy["psi_smoothing_count"]},
        "profiles": profiles,
    }

    # 3. Publish the immutable reference and its source audit
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(reference, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = run_record(cfg, run_id, "monitor_reference", version, "published",
                          model_version=version, reference_version=policy["reference_version"],
                          published_paths=[str(target)], reference_sha256=sha256(target),
                          input_refs={"model": reference["model_ref"],
                                      "feature_partitions": feature_refs,
                                      "score_partitions": score_refs,
                                      "monitoring_config": str(policy_path)})
    return {"status": "published", "reference_path": str(target),
            "reference_sha256": sha256(target), "manifest_path": str(manifest),
            "feature_rows": len(development), "score_rows": len(oot)}


def load_reference(config: dict, policy: dict, policy_path: Path,
                   bundle: dict, model_ref: Path) -> tuple[dict, Path]:
    path = reference_path(config, bundle["model_version"], policy["reference_version"])
    if not path.exists():
        raise FileNotFoundError(f"Frozen monitoring reference absent: {path}; run --build-reference")
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference["model_version"] != bundle["model_version"] or
            reference["model_sha256"] != sha256(model_ref) or
            reference["monitoring_config_sha256"] != sha256(policy_path) or
            reference["feature_columns"] != bundle["feature_columns"] or
            set(reference["profiles"]) != set(bundle["feature_columns"] + ["p_default"])):
        raise ValueError("Monitoring reference does not match model/config/feature contract")
    return reference, path


def _signal(rows: list[dict], context: dict, group: str, name: str,
            value: float | None, status: str = "valid", reason: str | None = None,
            n_reference: int | None = None, n_current: int | None = None) -> None:
    rows.append({**context, "signal_group": group, "signal_name": name,
                 "value": None if value is None else float(value), "status": status,
                 "reason": reason, "n_reference": n_reference, "n_current": n_current})


def _read_predictions(config: dict, application_month: str,
                      features: pd.DataFrame, feature_ref: dict,
                      model_version: str) -> tuple[pd.DataFrame, dict]:
    path = (location(config, "predictions") / model_version /
            application_month[:7] / "predictions.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Monitoring published predictions absent: {path}")
    predictions = pd.read_parquet(path)
    require_columns(predictions, PREDICTION_COLUMNS, "Monitoring predictions")
    if predictions.empty or predictions[KEYS].isna().any().any():
        raise ValueError("Monitoring predictions: empty cohort or null key")
    if predictions[PREDICTION_COLUMNS[4:]].isna().any().any():
        raise ValueError("Monitoring predictions: missing provenance field")
    if predictions.duplicated(KEYS + ["model_version"]).any():
        raise ValueError("Monitoring predictions: duplicate canonical keys")
    if not pd.to_datetime(predictions["prediction_as_of_date"]).dt.date.eq(
            month(application_month)).all():
        raise ValueError("Monitoring predictions: wrong historical prediction as-of")
    expected_ref = (f"{Path(feature_ref['path']).relative_to(config['root']).as_posix()}"
                    f"#sha256={feature_ref['sha256']}")
    stored_refs = predictions["feature_input_ref"].str.replace("\\", "/", regex=False)
    if not stored_refs.eq(expected_ref).all():
        raise ValueError("Monitoring predictions: feature input reference is stale")
    validate_predictions(predictions, features, model_version)
    return predictions, {"path": str(path), "sha256": sha256(path)}


def evaluate_labels(predictions: pd.DataFrame, application_month: str,
                    monitoring_as_of: date, config: dict) -> tuple[dict, pd.DataFrame, dict | None]:
    """Join only MOB-6 labels available at the explicit historical as-of."""
    target = config["project"]["target"]
    snapshot = add_months(month(application_month), target["horizon_months"])
    available = label_available_date(snapshot, target["reporting_lag_months"])
    expected = len(predictions) if available <= monitoring_as_of else 0
    result = {"snapshot_month": snapshot.isoformat(),
              "label_available_date": available.isoformat(),
              "n_predictions": len(predictions), "n_expected_eligible": expected,
              "n_matched_labels": 0, "label_coverage": None,
              "status": "pending_label" if expected == 0 else "incomplete",
              "reason": "label_not_mature" if expected == 0 else "label_partition_missing"}
    if expected == 0:
        return result, predictions.iloc[0:0].copy(), None
    path = partition(location(config, "gold_labels"), "gold_label_store",
                     snapshot.isoformat(), "parquet")
    if not path.exists():
        result["label_coverage"] = 0.0
        return result, predictions.iloc[0:0].copy(), None
    labels = pd.read_parquet(path)
    require_columns(labels, LABEL_COLUMNS, "Monitoring labels", exact=True)
    if (labels[["loan_id", "Customer_ID", "snapshot_date"]].isna().any().any() or
            labels.duplicated(["Customer_ID"]).any() or
            labels.duplicated(["loan_id"]).any()):
        raise ValueError("Monitoring labels: null or ambiguous loan/customer identity")
    if (not labels["label_def"].eq(target["label_def"]).all() or
            not labels["label"].isin([0, 1]).all() or
            not pd.to_datetime(labels["snapshot_date"]).dt.date.eq(snapshot).all()):
        raise ValueError("Monitoring labels: target, horizon or snapshot mismatch")
    if not set(labels["Customer_ID"]).issubset(set(predictions["Customer_ID"])):
        raise ValueError("Monitoring labels: loan/customer outside application cohort")
    joined = predictions.merge(labels, on="Customer_ID", how="left",
                               validate="one_to_one", indicator=True)
    matched = joined[joined["_merge"].eq("both")].drop(columns="_merge")
    n_matched = len(matched)
    result.update({"n_matched_labels": n_matched,
                   "label_coverage": n_matched / expected,
                   "status": "valid" if n_matched == expected else "partial_label",
                   "reason": None if n_matched == expected else "missing_eligible_labels"})
    return result, matched, {"path": str(path), "sha256": sha256_tree(path)}


def performance_signals(rows: list[dict], context: dict,
                        predictions: pd.DataFrame, matched: pd.DataFrame,
                        coverage: dict, calibration_bins: int) -> None:
    """Keep each metric's availability independent of other metrics."""
    state = coverage["status"]
    _signal(rows, context, "coverage", "label_coverage", coverage["label_coverage"],
            state, coverage["reason"], n_current=coverage["n_matched_labels"])
    if matched.empty:
        for name in METRICS:
            if name == "mean_predicted_probability":
                _signal(rows, context, "performance", name,
                        float(predictions["p_default"].mean()), "valid",
                        "all_published_predictions", n_current=len(predictions))
            else:
                _signal(rows, context, "performance", name, None,
                        state if state == "pending_label" else "unavailable",
                        coverage["reason"], n_current=0)
        return
    y = matched["label"].astype(int).to_numpy()
    probabilities = matched["p_default"].to_numpy(dtype=float)
    metric_values = {
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "actual_default_rate": float(np.mean(y)),
        "mean_predicted_probability": float(np.mean(probabilities)),
    }
    if len(set(y)) == 2:
        metric_values["roc_auc"] = float(roc_auc_score(y, probabilities))
        metric_values["average_precision"] = float(average_precision_score(y, probabilities))
    for name in METRICS:
        if name in metric_values:
            _signal(rows, context, "performance", name, metric_values[name], state,
                    coverage["reason"], n_current=len(matched))
        else:
            _signal(rows, context, "performance", name, None, "unavailable",
                    "single_class_labels", n_current=len(matched))
    indexes = np.minimum((probabilities * calibration_bins).astype(int),
                         calibration_bins - 1)
    for index in range(calibration_bins):
        chosen = indexes == index
        count = int(chosen.sum())
        for suffix, values in (("mean_score", probabilities), ("actual_rate", y)):
            _signal(rows, context, "calibration", f"bin_{index:02d}_{suffix}",
                    float(np.mean(values[chosen])) if count else None,
                    state if count else "unavailable",
                    coverage["reason"] if count else "empty_calibration_bin",
                    n_current=count)


def compute_month(application_month: str, monitoring_as_of: str,
                  config: dict | str | Path, run_id: str, bundle: dict,
                  model_ref: Path, reference: dict, reference_ref: Path,
                  policy: dict, policy_ref: Path) -> dict:
    """Compute one cohort and leave its checked table outside published paths."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    cohort = month(application_month)
    as_of = date.fromisoformat(monitoring_as_of)
    if as_of < cohort:
        raise ValueError("Monitoring as-of precedes application cohort")
    version = bundle["model_version"]
    ref_version = reference["reference_version"]

    # 1. Load and reconcile Gold Features with published predictions
    features, feature_ref = _read_feature_month(cfg, application_month, bundle)
    predictions, prediction_ref = _read_predictions(
        cfg, application_month, features, feature_ref, version)
    input_refs = {"features": feature_ref, "predictions": prediction_ref,
                  "model": {"path": str(model_ref), "sha256": sha256(model_ref)},
                  "reference": {"path": str(reference_ref), "sha256": sha256(reference_ref)},
                  "monitoring_config": {"path": str(policy_ref), "sha256": sha256(policy_ref)}}
    context = {"application_month": cohort, "model_version": version,
               "monitoring_as_of": as_of, "reference_version": ref_version,
               "n_predictions": len(predictions),
               "n_expected_eligible": 0, "n_matched_labels": 0,
               "label_coverage": None, "run_id": run_id,
               "feature_ref": str(feature_ref["path"]),
               "prediction_ref": str(prediction_ref["path"]),
               "label_ref": None,
               "model_ref": str(model_ref), "reference_ref": str(reference_ref),
               "monitoring_config_ref": str(policy_ref)}
    rows: list[dict] = []

    # 2. Audit batch quality, contract-aware missingness and fixed-bin PSI
    _signal(rows, context, "quality", "expected_applications", len(features),
            n_current=len(features))
    _signal(rows, context, "quality", "published_predictions", len(predictions),
            n_current=len(predictions))
    _signal(rows, context, "quality", "prediction_coverage", 1.0,
            n_current=len(predictions))
    for name in ("minimum_score", "mean_score", "maximum_score"):
        value = {"minimum_score": predictions["p_default"].min(),
                 "mean_score": predictions["p_default"].mean(),
                 "maximum_score": predictions["p_default"].max()}[name]
        _signal(rows, context, "quality", name, float(value),
                n_current=len(predictions))
    nullable = set(bundle.get("nullable_numeric_columns", []))
    for name in bundle["numeric_columns"]:
        missing = int(features[name].isna().sum())
        if missing and name not in nullable:
            raise ValueError(f"Monitoring: non-nullable predictor missing: {name}")
        _signal(rows, context, "missingness", name, missing / len(features),
                reason="allowed_numeric_nan" if missing else None,
                n_current=len(features))
    min_size = policy["minimum_stability_sample_size"]
    for name in bundle["feature_columns"] + ["p_default"]:
        values = predictions["p_default"] if name == "p_default" else features[name]
        profile = reference["profiles"][name]
        n_reference = profile["n_reference"]
        if min(n_reference, len(values)) < min_size:
            _signal(rows, context, "stability", name, None, "unavailable",
                    "sample_below_minimum", n_reference, len(values))
            continue
        psi = population_stability_index(values, profile,
                                         policy["psi_smoothing_count"])
        status = "review" if psi >= policy["psi_review_threshold"] else "valid"
        _signal(rows, context, "stability", name, psi, status,
                "psi_threshold_review" if status == "review" else None,
                n_reference, len(values))

    # 3. Evaluate only labels available at this as-of, metric by metric
    coverage, matched, label_ref = evaluate_labels(predictions, application_month,
                                                   as_of, cfg)
    input_refs["labels"] = label_ref
    context.update({"n_expected_eligible": coverage["n_expected_eligible"],
                    "n_matched_labels": coverage["n_matched_labels"],
                    "label_coverage": coverage["label_coverage"],
                    "label_ref": str(label_ref["path"]) if label_ref else None})
    for row in rows:
        row.update({name: context[name] for name in
                    ("n_expected_eligible", "n_matched_labels", "label_coverage",
                     "label_ref")})
    performance_signals(rows, context, predictions, matched, coverage,
                        policy["calibration_bins"])
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_monitoring_table(table, application_month, monitoring_as_of,
                              version, ref_version)

    # 4. Stage one complete cohort table for independent publication
    staging = (cfg["root"] / ".staging" / "monitoring" / run_id / version /
               ref_version / monitoring_as_of / application_month[:7] /
               "monitoring.parquet")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise FileExistsError(f"Monitoring staging already exists: {staging}")
    table.to_parquet(staging, index=False)
    return {"run_id": run_id, "status": "staged", "application_month": application_month,
            "model_version": version, "monitoring_as_of": monitoring_as_of,
            "reference_version": ref_version, "staging_paths": [str(staging)],
            "input_refs": input_refs, "output_counts": {"rows": len(table), **coverage}}


def validate_monitoring_table(table: pd.DataFrame, application_month: str,
                              monitoring_as_of: str, model_version: str,
                              reference_version: str) -> None:
    require_columns(table, OUTPUT_COLUMNS, "Gold monitoring", exact=True)
    if table.empty or table.duplicated(["signal_group", "signal_name"]).any():
        raise ValueError("Gold monitoring: empty or duplicate signal key")
    expected = {"application_month": month(application_month),
                "monitoring_as_of": date.fromisoformat(monitoring_as_of),
                "model_version": model_version, "reference_version": reference_version}
    for name, value in expected.items():
        actual = pd.to_datetime(table[name]).dt.date if name.endswith(("month", "as_of")) else table[name]
        if not actual.eq(value).all():
            raise ValueError(f"Gold monitoring: wrong {name}")
    values = table["value"].dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Gold monitoring: non-finite signal value")
    if not table["status"].isin(["valid", "review", "pending_label",
                                "partial_label", "incomplete", "unavailable"]).all():
        raise ValueError("Gold monitoring: invalid signal status")
    if ((table["n_predictions"] <= 0).any() or
            (table["n_expected_eligible"] < 0).any() or
            (table["n_matched_labels"] < 0).any() or
            (table["n_matched_labels"] > table["n_expected_eligible"]).any()):
        raise ValueError("Gold monitoring: invalid cohort coverage counts")
    mature = table["n_expected_eligible"] > 0
    if (table.loc[mature, "label_coverage"].sub(
            table.loc[mature, "n_matched_labels"] /
            table.loc[mature, "n_expected_eligible"]).abs() > 1e-12).any():
        raise ValueError("Gold monitoring: label coverage differs from counts")
    if table.loc[~mature, "label_coverage"].notna().any():
        raise ValueError("Gold monitoring: pending cohort has numeric label coverage")


def publish_month(staging_ref: dict, config: dict | str | Path) -> dict:
    cfg = load_config(config) if not isinstance(config, dict) else config
    staged = Path(staging_ref["staging_paths"][0])
    table = pd.read_parquet(staged)
    scope = staging_ref["application_month"]
    version = staging_ref["model_version"]
    as_of = staging_ref["monitoring_as_of"]
    ref_version = staging_ref["reference_version"]

    # 1. Validate staged schema/scope and unchanged source hashes
    validate_monitoring_table(table, scope, as_of, version, ref_version)
    refs = staging_ref["input_refs"]
    if len(table) != staging_ref["output_counts"]["rows"]:
        raise ValueError("Monitoring staging row count changed")
    for name in ("features", "predictions", "model", "reference", "monitoring_config"):
        ref = refs[name]
        path = Path(ref["path"])
        actual = sha256_tree(path) if path.is_dir() else sha256(path)
        if actual != ref["sha256"]:
            raise ValueError(f"Monitoring {name} changed after computation")
    if refs["labels"]:
        ref = refs["labels"]
        if sha256_tree(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("Monitoring labels changed after computation")
    elif staging_ref["output_counts"]["n_expected_eligible"]:
        snapshot = staging_ref["output_counts"]["snapshot_month"]
        label_path = partition(location(cfg, "gold_labels"), "gold_label_store",
                               snapshot, "parquet")
        if label_path.exists():
            raise ValueError("Monitoring labels appeared after computation")

    # 2. Replace only this cohort/model/reference/as-of partition
    target = monitoring_path(cfg, version, ref_version, as_of, scope)
    publish_partition(staged, target)
    _remove_empty_staging_parents(staged, cfg)
    manifest = run_record(cfg, staging_ref["run_id"], "monitor", scope, "published",
                          application_month=scope, model_version=version,
                          monitoring_as_of=as_of, reference_version=ref_version,
                          input_refs=refs, output_counts=staging_ref["output_counts"],
                          published_paths=[str(target)])
    return {"run_id": staging_ref["run_id"], "status": "published",
            "application_month": scope, "model_version": version,
            "monitoring_as_of": as_of, "reference_version": ref_version,
            "published_paths": [str(target)], "manifest_path": str(manifest),
            "output_counts": staging_ref["output_counts"]}


def _series(table: pd.DataFrame, group: str, name: str,
            cohort_months: list[str]) -> np.ndarray:
    subset = table[(table["signal_group"] == group) &
                   (table["signal_name"] == name)]
    values = {pd.Timestamp(row.application_month).strftime("%Y-%m"): row.value
              for row in subset.itertuples()}
    return np.array([values.get(value, np.nan) for value in cohort_months], dtype=float)


def render_report(config: dict, run_id: str, model_version: str,
                  reference_version: str, monitoring_as_of: str) -> dict:
    """Publish charts from all currently published cohorts at one exact as-of."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = (location(config, "monitoring") / model_version / reference_version /
            monitoring_as_of)
    paths = sorted(root.glob("*/monitoring.parquet"))
    if not paths:
        raise ValueError("Monitoring report: no published cohort tables")
    tables = [pd.read_parquet(path).astype({"value": "float64",
                                           "label_coverage": "float64"}) for path in paths]
    table = pd.concat(tables, ignore_index=True)
    cohort_months = sorted(pd.to_datetime(table["application_month"]).dt.strftime("%Y-%m").unique())
    title = (f"Model: {model_version}\nReference: {reference_version} | "
             f"As-of: {monitoring_as_of}")
    staging = (config["root"] / ".staging" / "monitoring_reports" / run_id /
               model_version / reference_version / monitoring_as_of)
    if staging.exists():
        raise FileExistsError(f"Monitoring report staging already exists: {staging}")
    staging.mkdir(parents=True)
    x = np.arange(len(cohort_months))

    # 1. Plot metric support and actual-versus-predicted default rates
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, name in zip(axes.flat, METRICS[:4]):
        values = _series(table, "performance", name, cohort_months)
        axis.plot(x, values, marker="o")
        axis.set_title(name.replace("_", " ").title())
        axis.grid(alpha=0.2)
        axis.set_xticks(x, cohort_months, rotation=45)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(staging / "performance.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    for name, label in (("actual_default_rate", "Actual default rate"),
                        ("mean_predicted_probability", "Mean predicted probability")):
        axis.plot(x, _series(table, "performance", name, cohort_months),
                  marker="o", label=label)
    axis.set_xticks(x, cohort_months, rotation=45)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate / probability")
    axis.set_title(title, fontsize=10)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(staging / "default_rate_vs_prediction.png", dpi=150)
    plt.close(fig)

    # 2. Plot fixed-reference PSI and eligible-label coverage
    stability = table[table["signal_group"].eq("stability")]
    names = sorted(stability["signal_name"].unique())
    matrix = np.vstack([_series(table, "stability", name, cohort_months)
                        for name in names])
    fig, axis = plt.subplots(figsize=(max(10, len(cohort_months) * 1.3), 12))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest",
                        cmap="YlOrRd", vmin=0)
    axis.set_yticks(np.arange(len(names)), names, fontsize=7)
    axis.set_xticks(x, cohort_months, rotation=45)
    axis.set_title(f"PSI | {title}", fontsize=10)
    fig.colorbar(image, ax=axis, label="PSI (fixed bins)")
    fig.tight_layout()
    fig.savefig(staging / "stability.png", dpi=150)
    plt.close(fig)

    coverage = table[table["signal_group"].eq("coverage")].copy()
    coverage["month"] = pd.to_datetime(coverage["application_month"]).dt.strftime("%Y-%m")
    coverage = coverage.set_index("month").reindex(cohort_months)
    fig, axis = plt.subplots(figsize=(11, 5))
    expected = coverage["n_expected_eligible"].fillna(0).to_numpy(dtype=int)
    matched = coverage["n_matched_labels"].fillna(0).to_numpy(dtype=int)
    pending = coverage["status"].eq("pending_label").to_numpy()
    axis.bar(x, matched, label="Matched labels")
    axis.bar(x, expected - matched, bottom=matched, label="Eligible, unmatched")
    for index, is_pending in enumerate(pending):
        if is_pending:
            axis.annotate("pending", (index, 0), xytext=(0, 8),
                          textcoords="offset points", ha="center", rotation=90)
    axis.set_xticks(x, cohort_months, rotation=45)
    axis.set_ylabel("Applications")
    axis.set_title(f"Label coverage | {title}", fontsize=10)
    axis.legend()
    fig.tight_layout()
    fig.savefig(staging / "label_coverage.png", dpi=150)
    plt.close(fig)

    # 3. Show calibration only for bins with observed mature labels
    calibration = table[table["signal_group"].eq("calibration")].copy()
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="Perfect calibration")
    for cohort in cohort_months:
        subset = calibration[pd.to_datetime(calibration["application_month"]).dt.strftime(
            "%Y-%m").eq(cohort)]
        if subset.empty:
            continue
        means = {row.signal_name.replace("_mean_score", ""): (row.value, row.n_current)
                 for row in subset.itertuples() if row.signal_name.endswith("_mean_score")}
        rates = {row.signal_name.replace("_actual_rate", ""): row.value
                 for row in subset.itertuples() if row.signal_name.endswith("_actual_rate")}
        points = [(means[key][0], rates[key], means[key][1]) for key in sorted(means)
                  if key in rates and pd.notna(means[key][0]) and pd.notna(rates[key])]
        if points:
            xs, ys, counts = zip(*points)
            line, = axis.plot(xs, ys, label=cohort)
            axis.scatter(xs, ys, s=[min(180, 15 + 2 * count) for count in counts],
                         color=line.get_color())
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
             ylabel="Observed default rate")
    axis.set_title(f"Calibration | {title}", fontsize=9)
    axis.legend(fontsize=8)
    fig.text(0.5, 0.01, "Marker area reflects bin sample size; rare bins are noisy.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(staging / "calibration.png", dpi=150)
    plt.close(fig)

    # 4. Publish the complete chart set and a concise machine-readable summary
    summary = {"run_id": run_id, "model_version": model_version,
               "reference_version": reference_version, "monitoring_as_of": monitoring_as_of,
               "cohorts": cohort_months,
               "gold_refs": [{"path": str(path), "sha256": sha256(path)} for path in paths],
               "coverage": [{"application_month": value,
                             "n_predictions": int(coverage.loc[value, "n_predictions"]),
                             "n_expected_eligible": int(coverage.loc[value, "n_expected_eligible"]),
                             "n_matched_labels": int(coverage.loc[value, "n_matched_labels"]),
                             "status": str(coverage.loc[value, "status"])}
                            for value in cohort_months]}
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    target = (location(config, "monitoring_reports") / model_version /
              reference_version / monitoring_as_of)
    publish_partition(staging, target)
    _remove_empty_staging_parents(staging, config)
    manifest = run_record(config, run_id, "monitor_report", monitoring_as_of,
                          "published", model_version=model_version,
                          reference_version=reference_version,
                          monitoring_as_of=monitoring_as_of,
                          input_refs=summary["gold_refs"],
                          output_counts={"cohorts": len(cohort_months)},
                          published_paths=[str(target)])
    return {"report_path": str(target), "manifest_path": str(manifest),
            "cohorts": cohort_months}


def run(config: dict | str | Path, run_id: str, monitoring_as_of: str,
        application_month: str | None = None, start: str | None = None,
        end_exclusive: str | None = None, model_version: str | None = None,
        model_path: Path | None = None, monitoring_config: Path | str | None = None,
        reference_version: str | None = None) -> dict:
    """Monitor requested cohorts using published scores and explicit as-of."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    as_of = date.fromisoformat(monitoring_as_of).isoformat()
    if application_month and (start or end_exclusive):
        raise ValueError("Select one application month or one half-open range")
    if not application_month and ((start is None) != (end_exclusive is None)):
        raise ValueError("Monitoring range requires start and end-exclusive")
    requested = ([month(application_month).isoformat()] if application_month else
                 months(*(start, end_exclusive) if start else
                        cfg["project"]["splits"]["inference"]))

    # 1. Resolve one model, reference and historical as-of for the whole run
    try:
        bundle, model_ref = resolve_model(cfg, model_version, model_path)
        policy, policy_ref = load_monitoring_config(cfg, monitoring_config)
        if reference_version:
            policy = {**policy, "reference_version": _reference_version(reference_version)}
        reference, reference_ref = load_reference(cfg, policy, policy_ref, bundle, model_ref)
    except Exception as error:
        run_record(cfg, run_id, "monitor", "resolve", "failed",
                   monitoring_as_of=as_of, model_version=model_version,
                   reference_version=reference_version, error=str(error))
        raise

    # 2. Compute, validate and publish each cohort independently
    results = []
    for value in requested:
        try:
            staged = compute_month(value, as_of, cfg, run_id, bundle, model_ref,
                                   reference, reference_ref, policy, policy_ref)
            results.append(publish_month(staged, cfg))
        except Exception as error:
            run_record(cfg, run_id, "monitor", value, "failed",
                       application_month=value, monitoring_as_of=as_of,
                       model_version=bundle["model_version"],
                       reference_version=reference["reference_version"], error=str(error))
            raise

    # 3. Refresh charts for all published cohorts at this exact as-of
    try:
        report = render_report(cfg, run_id, bundle["model_version"],
                               reference["reference_version"], as_of)
    except Exception as error:
        run_record(cfg, run_id, "monitor_report", as_of, "failed",
                   monitoring_as_of=as_of, model_version=bundle["model_version"],
                   reference_version=reference["reference_version"], error=str(error))
        raise
    return {"run_id": run_id, "status": "published", "cohorts": results,
            "report": report, "model_version": bundle["model_version"],
            "reference_version": reference["reference_version"],
            "monitoring_as_of": as_of}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--monitoring-config", type=Path)
    parser.add_argument("--build-reference", action="store_true")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--application-month")
    scope.add_argument("--start")
    parser.add_argument("--end-exclusive")
    parser.add_argument("--monitoring-as-of")
    parser.add_argument("--model-version")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--reference-version")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg = args.config or load_config()["config_path"]
    if args.build_reference:
        if args.application_month or args.start or args.end_exclusive or args.monitoring_as_of:
            parser.error("Reference build has no cohort or monitoring as-of")
        result = build_reference(cfg, args.run_id, args.model_version,
                                 args.model_path, args.monitoring_config,
                                 args.reference_version)
    else:
        if not args.monitoring_as_of:
            parser.error("--monitoring-as-of is required")
        if args.start and not args.end_exclusive:
            parser.error("--start requires --end-exclusive")
        if args.end_exclusive and not args.start:
            parser.error("--end-exclusive requires --start")
        result = run(cfg, args.run_id, args.monitoring_as_of, args.application_month,
                     args.start, args.end_exclusive, args.model_version,
                     args.model_path, args.monitoring_config, args.reference_version)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
