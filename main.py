"""Local runner for the implemented data, training, and inference stages."""

import argparse
import json
from pathlib import Path

from pipelines import build_features, build_labels, inference, train
from pipelines.utils.config import load_config


def run_current_flow(config: Path, run_id: str) -> dict:
    """Run current A-D capabilities without implying monitoring or Airflow."""
    cfg = load_config(config)
    project = cfg["project"]
    # 1. Build the fixed baseline feature and label partitions
    features = build_features.run_range(*project["source_coverage"]["applications"],
                                        cfg, run_id)
    labels = build_labels.run_range("2023-01-01", "2025-01-01", cfg, run_id)
    # 2. Refit, validate, and release a distinct version
    model = train.run(cfg, run_id, publish=True)
    # 3. Score and publish the fixed inference cohorts
    predictions = inference.run(cfg, run_id)
    return {"run_id": run_id, "features": features, "labels": labels,
            "model": model, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("command", choices=["bootstrap-current"])
    args = parser.parse_args()
    print(json.dumps(run_current_flow(args.config or load_config()["config_path"],
                                      args.run_id), indent=2))


if __name__ == "__main__":
    main()
