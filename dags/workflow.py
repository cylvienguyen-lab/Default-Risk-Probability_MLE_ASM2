"""Small, testable task callables shared by the two Airflow DAGs."""

import hashlib
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from pipelines import build_features, build_labels, inference, monitor, train
from pipelines.utils.artifacts import read_released_model, run_record, sha256
from pipelines.utils.config import add_months, load_config, location, month, months


def task_run_id(dag_run_id: str, task_name: str, attempt: int,
                map_index: int = -1) -> str:
    """Identify one mapped task attempt without unsafe path characters."""
    if attempt < 1 or map_index < -1:
        raise ValueError("Task attempt or map index is invalid")
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", dag_run_id)[:56]
    name = re.sub(r"[^A-Za-z0-9_-]", "_", task_name)
    digest = hashlib.sha256(dag_run_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}_{name}_m{map_index}_a{attempt}"


def bootstrap_scope(config_path: str, dag_run_id: str,
                    monitoring_as_of: str | None = None) -> dict:
    cfg = load_config(config_path)
    project = cfg["project"]
    fit_as_of = month(project["final_fit_as_of"])
    inference_months = months(*project["splits"]["inference"])
    delay = project["target"]["horizon_months"] + project["target"]["reporting_lag_months"]
    last_mature_as_of = add_months(month(inference_months[-1]), delay)
    as_of = month(monitoring_as_of or last_mature_as_of.isoformat())
    if as_of < last_mature_as_of:
        raise ValueError("Bootstrap monitoring as-of must mature the last inference cohort")
    training_months = [value for name in ("development", "test", "oot")
                       for value in months(*project["splits"][name])]
    feature_months = sorted(set(training_months + inference_months))
    training_snapshots = sorted({add_months(month(value), project["target"]["horizon_months"])
                                 .isoformat() for value in training_months})
    monitoring_snapshots = sorted({add_months(month(value), project["target"]["horizon_months"])
                                   .isoformat() for value in inference_months})
    source_end = month(project["source_coverage"]["outcomes"][1])
    if any(month(value) >= source_end for value in
           training_snapshots + monitoring_snapshots):
        raise ValueError("Bootstrap label snapshots exceed declared outcome coverage")
    if fit_as_of > month(inference_months[0]):
        raise ValueError("Final fit as-of is after first inference cohort")
    return {"config_path": str(cfg["config_path"]), "run_id": dag_run_id,
            "model_version": cfg["model"]["model_version"],
            "fit_as_of": fit_as_of.isoformat(), "monitoring_as_of": as_of.isoformat(),
            "feature_months": feature_months,
            "training_label_snapshots": training_snapshots,
            "monitoring_label_snapshots": monitoring_snapshots,
            "inference_months": inference_months}


def monthly_scope(config_path: str, dag_run_id: str, interval_end: str,
                  application_month: str | None = None,
                  monitoring_as_of: str | None = None,
                  model_version: str | None = None) -> dict:
    cfg = load_config(config_path)
    as_of = month(monitoring_as_of or date.fromisoformat(interval_end).isoformat())
    current = month(application_month or add_months(as_of, -1).isoformat())
    if current != add_months(as_of, -1):
        raise ValueError("Monthly application cohort must immediately precede monitoring as-of")
    start, end = map(month, cfg["project"]["splits"]["inference"])
    if not start <= current < end:
        raise ValueError("Monthly cohort is outside configured inference range")
    source_start, source_end = map(month, cfg["project"]["source_coverage"]["applications"])
    if not source_start <= current < source_end:
        raise ValueError("Monthly cohort is outside declared application source coverage")
    target = cfg["project"]["target"]
    mature = add_months(as_of, -(target["horizon_months"] + target["reporting_lag_months"]))
    eligible = start <= mature < end
    snapshot = add_months(as_of, -target["reporting_lag_months"])
    if eligible and snapshot >= month(cfg["project"]["source_coverage"]["outcomes"][1]):
        raise ValueError("Eligible outcome snapshot exceeds declared source coverage")
    cohorts = [current.isoformat()]
    if eligible and mature != current:
        cohorts.append(mature.isoformat())
    return {"config_path": str(cfg["config_path"]), "run_id": dag_run_id,
            "application_month": current.isoformat(),
            "monitoring_as_of": as_of.isoformat(),
            "label_snapshot": snapshot.isoformat() if eligible else None,
            "monitor_cohorts": cohorts, "model_version": model_version}


def build_feature(application_month: str, config_path: str, run_id: str) -> dict:
    return build_features.run_month(application_month, config_path, run_id)


def build_label(snapshot_month: str, config_path: str, run_id: str) -> dict:
    return build_labels.run_snapshot(snapshot_month, config_path, run_id)


def validate_gold(config_path: str, run_id: str) -> dict:
    cfg = load_config(config_path)
    assembled, refs = train.assemble_datasets(cfg)
    counts = {name: int(assembled["split"].eq(name).sum())
              for name in ("development", "test", "oot")}
    manifest = run_record(cfg, run_id, "validate_gold", "training", "validated",
                          input_refs=refs, output_counts=counts)
    return {"status": "validated", "manifest_path": str(manifest), "counts": counts}


def train_candidate(config_path: str, run_id: str, fit_as_of: str) -> dict:
    return train.run(config_path, run_id, fit_as_of=fit_as_of, publish=False)


def publish_model(candidate: dict, config_path: str, run_id: str) -> dict:
    return train.publish_candidate(Path(candidate["staging_paths"][0]), config_path,
                                   run_id, candidate["manifest_path"])


def released_model(config_path: str, requested_version: str | None = None) -> dict:
    cfg = load_config(config_path)
    pointer = read_released_model(location(cfg, "model_bank"))
    if requested_version and pointer["model_version"] != requested_version:
        raise ValueError("Requested version is not the current released model")
    bundle, path = inference.resolve_model(cfg, pointer["model_version"],
                                           Path(pointer["artifact_path"]))
    return {"model_version": bundle["model_version"], "model_path": str(path),
            "model_sha256": sha256(path), "manifest_path": pointer["manifest_path"]}


def released_from_publication(publication: dict, config_path: str) -> dict:
    return released_model(config_path, publication["model_version"])


def score(application_month: str, config_path: str, run_id: str, model: dict) -> dict:
    cfg = load_config(config_path)
    bundle, path = inference.resolve_model(cfg, model["model_version"],
                                           Path(model["model_path"]))
    if sha256(path) != model["model_sha256"]:
        raise ValueError("Released model changed between DAG tasks")
    return inference.score_month(application_month, cfg, run_id, bundle, path)


def publish_prediction(staging: dict, config_path: str) -> dict:
    return inference.publish_month(staging, config_path)


def ensure_reference(config_path: str, run_id: str, model: dict) -> dict:
    cfg = load_config(config_path)
    policy, policy_path = monitor.load_monitoring_config(cfg)
    bundle, model_path = inference.resolve_model(cfg, model["model_version"],
                                                 Path(model["model_path"]))
    if sha256(model_path) != model["model_sha256"]:
        raise ValueError("Released model changed before reference resolution")
    path = monitor.reference_path(cfg, bundle["model_version"], policy["reference_version"])
    if not path.exists():
        monitor.build_reference(cfg, run_id, model_version=bundle["model_version"],
                                model_path=model_path)
    reference, reference_path = monitor.load_reference(
        cfg, policy, policy_path, bundle, model_path)
    return {"reference_version": reference["reference_version"],
            "reference_path": str(reference_path),
            "reference_sha256": sha256(reference_path)}


def compute_monitoring(application_month: str, monitoring_as_of: str,
                       config_path: str, run_id: str, model: dict,
                       reference: dict) -> dict:
    cfg = load_config(config_path)
    bundle, model_path = inference.resolve_model(cfg, model["model_version"],
                                                 Path(model["model_path"]))
    policy, policy_path = monitor.load_monitoring_config(cfg)
    if policy["reference_version"] != reference["reference_version"]:
        raise ValueError("Monitoring reference version changed between tasks")
    frozen, ref_path = monitor.load_reference(cfg, policy, policy_path,
                                              bundle, model_path)
    if sha256(ref_path) != reference["reference_sha256"]:
        raise ValueError("Monitoring reference changed between DAG tasks")
    return monitor.compute_month(application_month, monitoring_as_of, cfg, run_id,
                                 bundle, model_path, frozen, ref_path, policy,
                                 policy_path)


def publish_monitoring(staging: dict, config_path: str) -> dict:
    return monitor.publish_month(staging, config_path)


def report(config_path: str, run_id: str, monitoring_as_of: str,
           model: dict, reference: dict) -> dict:
    return monitor.render_report(load_config(config_path), run_id,
                                 model["model_version"],
                                 reference["reference_version"], monitoring_as_of)


def ready_cohorts(expected: list[str], predictions: list[dict],
                  labels: list[dict] | dict, config_path: str | None = None,
                  model_version: str | None = None,
                  expected_label_snapshots: list[str] | None = None) -> list[str]:
    """Gate monitoring on actual published scores and required label jobs."""
    published = {item["application_month"] for item in predictions
                 if item["status"] == "published"}
    if published - set(expected):
        raise ValueError("Published predictions exceed monitoring scope")
    absent = set(expected) - published
    if absent and (not config_path or not model_version):
        raise ValueError("Published prediction cohorts do not match monitoring scope")
    if absent:
        cfg = load_config(config_path)
        for value in absent:
            path = location(cfg, "predictions") / model_version / value[:7] / "predictions.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Historical published predictions absent: {path}")
    label_results = [labels] if isinstance(labels, Mapping) else list(labels)
    if any(item["status"] != "published" for item in label_results
           if item.get("status") != "not_eligible"):
        raise ValueError("Required monitoring labels are not published")
    if expected_label_snapshots is not None:
        actual = {item["outcome_snapshot_month"] for item in label_results
                  if item["status"] == "published"}
        if actual != set(expected_label_snapshots):
            raise ValueError("Published label snapshots do not match monitoring scope")
    return expected
