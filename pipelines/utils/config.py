"""Load the three project contracts independently of the working directory."""

import json
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "project_config.json"


def load_config(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as stream:
        project = json.load(stream)
    root = path.parent.parent
    with (path.parent / project["feature_config"]).open(encoding="utf-8") as stream:
        feature = json.load(stream)
    with (path.parent / project["model_config"]).open(encoding="utf-8") as stream:
        model = json.load(stream)
    return {"project": project, "feature": feature, "model": model,
            "root": root, "config_path": path}


def location(config: dict, key: str) -> Path:
    return (config["root"] / config["project"]["paths"][key]).resolve()


def month(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.day != 1:
        raise ValueError(f"Expected first day of month: {value}")
    return parsed


def add_months(value: date, count: int) -> date:
    index = value.year * 12 + value.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def months(start: str, end_exclusive: str) -> list[str]:
    current, end = month(start), month(end_exclusive)
    if current >= end:
        raise ValueError("Start month must precede exclusive end month")
    result = []
    while current < end:
        result.append(current.isoformat())
        current = add_months(current, 1)
    return result


def partition(directory: Path, prefix: str, month_value: str, extension: str) -> Path:
    return directory / f"{prefix}_{month(month_value).strftime('%Y_%m_%d')}.{extension}"
