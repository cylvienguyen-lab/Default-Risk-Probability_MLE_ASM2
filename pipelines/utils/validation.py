"""Reusable data, temporal, model, and publication checks."""

from datetime import date

import numpy as np
import pandas as pd

from pipelines.utils.config import add_months, month


def require_columns(frame, columns, stage: str, exact: bool = False) -> None:
    actual = list(frame.columns)
    missing = set(columns) - set(actual)
    extra = set(actual) - set(columns) if exact else set()
    if missing or extra or (exact and actual != list(columns)):
        raise ValueError(f"{stage}: schema mismatch; missing={sorted(missing)}, "
                         f"extra={sorted(extra)}, order_matches={actual == list(columns)}")


def unique_keys(frame, keys: list[str], stage: str) -> None:
    import pyspark.sql.functions as F

    if frame.filter(F.expr(" OR ".join(f"`{key}` IS NULL" for key in keys))).limit(1).count():
        raise ValueError(f"{stage}: null key in {keys}")
    if frame.groupBy(*keys).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError(f"{stage}: duplicate key in {keys}")


def audit_invalid_to_null(raw, cleaned, columns: dict[str, str], stage: str) -> dict:
    """Count present raw values intentionally or unexpectedly null after cleaning."""
    import pyspark.sql.functions as F

    keys = ["Customer_ID", "snapshot_date"]
    joined = raw.alias("raw").join(cleaned.alias("clean"), on=keys, how="inner")
    if joined.count() != raw.count() or joined.count() != cleaned.count():
        raise ValueError(f"{stage}: cannot reconcile Bronze and Silver rows for null audit")
    expressions = [F.sum(F.when(
        F.col(f"raw.`{raw_name}`").isNotNull() &
        (F.trim(F.col(f"raw.`{raw_name}`").cast("string")) != "") &
        F.col(f"clean.`{clean_name}`").isNull(), 1).otherwise(0)).alias(clean_name)
                   for raw_name, clean_name in columns.items()]
    values = joined.agg(*expressions).first().asDict()
    return {name: int(value or 0) for name, value in values.items()}


def validate_feature_partition(frame, feature_config: dict, application_month: str) -> dict:
    import pyspark.sql.functions as F

    columns = feature_config["gold_columns"]
    require_columns(frame, columns, "Gold features", exact=True)
    expected_types = feature_config["gold_types"]
    for field in frame.schema.fields:
        actual = field.dataType.typeName()
        expected = expected_types[field.name]
        if actual not in ({"integer", "long"} if expected == "int" else
                          {"double", "float"} if expected == "double" else {expected}):
            raise TypeError(f"Gold features: {field.name} expected {expected}, got {actual}")
    unique_keys(frame, ["Customer_ID", "application_date"], "Gold features")
    non_nullable = [column for group in feature_config["predictors"].values()
                    for column in group["numerical"]["non_nullable"] + group["categorical"]]
    non_nullable += ["unusual_discrete_value_flag", "loan_info_inconsistent_flag",
                     "no_loan_flag", "clickstream_months_available",
                     "clickstream_history_available"]
    if frame.filter(F.expr(" OR ".join(f"`{name}` IS NULL" for name in non_nullable))).limit(1).count():
        raise ValueError("Gold features: null in contract non-nullable field")
    if frame.filter((~F.col("clickstream_history_available").isin(0, 1)) |
                    (~F.col("clickstream_months_available").between(0, 3))).limit(1).count():
        raise ValueError("Gold features: invalid Clickstream availability flags")
    floating = [field.name for field in frame.schema.fields
                if field.dataType.typeName() in ("float", "double")]
    invalid_floats = F.lit(False)
    for name in floating:
        invalid_floats = (invalid_floats | F.isnan(name) |
                          (F.abs(F.col(name)) == float("inf")))
    if frame.filter(invalid_floats).limit(1).count():
        raise ValueError("Gold features: non-finite numeric value")
    expected = month(application_month)
    if frame.filter(F.col("application_date") != F.lit(expected)).limit(1).count():
        raise ValueError("Gold features: row outside application month")
    count = frame.count()
    if count == 0:
        raise ValueError(f"Gold features: empty application month {application_month}")
    forbidden = {"Name", "SSN", "label", "dpd", "snapshot_date"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Gold features: forbidden fields {sorted(forbidden)}")
    return {"rows": count, "columns": len(columns)}


def validate_label_partition(frame, snapshot_month: str, label_def: str) -> dict:
    import pyspark.sql.functions as F

    require_columns(frame, ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"],
                    "Gold labels", exact=True)
    unique_keys(frame, ["loan_id", "snapshot_date"], "Gold labels")
    unique_keys(frame, ["Customer_ID", "snapshot_date"], "Gold labels")
    if frame.filter((F.col("snapshot_date") != F.lit(month(snapshot_month))) |
                    (F.col("label_def") != label_def) |
                    F.col("label").isNull() | (~F.col("label").isin(0, 1))).limit(1).count():
        raise ValueError(f"Gold labels: invalid target or scope for {snapshot_month}")
    return {"rows": frame.count()}


def label_available_date(snapshot_date: date, reporting_lag_months: int = 1) -> date:
    return add_months(snapshot_date, reporting_lag_months)


def split_name(application_date: date, splits: dict) -> str | None:
    for name, (start, end) in splits.items():
        if date.fromisoformat(start) <= application_date < date.fromisoformat(end):
            return name
    return None


def validate_training_stores(features: pd.DataFrame, labels: pd.DataFrame,
                             project: dict, required_splits: tuple[str, ...] =
                             ("development", "test", "oot")) -> pd.DataFrame:
    keys = ["Customer_ID", "application_date"]
    if features[keys].isna().any().any() or features.duplicated(keys).any():
        raise ValueError("Training features: null or duplicate application key")
    if features.duplicated(["Customer_ID"]).any():
        raise ValueError("Training: multiple applications per customer are unsupported")
    required = ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]
    require_columns(labels, required, "Training labels", exact=True)
    if labels["Customer_ID"].isna().any() or labels.duplicated(["Customer_ID"]).any():
        raise ValueError("Training: ambiguous label Customer_ID")
    if labels["label"].isna().any() or not labels["label"].isin([0, 1]).all():
        raise ValueError("Training: invalid binary label")
    if not labels["label_def"].eq(project["target"]["label_def"]).all():
        raise ValueError("Training: label definition mismatch")
    result = features.merge(labels, on="Customer_ID", how="left", validate="one_to_one",
                            indicator=True)
    result["application_date"] = pd.to_datetime(result["application_date"])
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"])
    result["split"] = result["application_date"].dt.date.map(
        lambda value: split_name(value, project["splits"]))
    expected = result["application_date"].dt.date.map(
        lambda value: add_months(value, project["target"]["horizon_months"]))
    labeled = result["_merge"].eq("both")
    if not (result.loc[labeled, "snapshot_date"].dt.date.to_numpy() ==
            expected.loc[labeled].to_numpy()).all():
        raise ValueError("Training: label horizon mismatch")
    for name in required_splits:
        subset = result[result["split"].eq(name)]
        if subset.empty or not subset["_merge"].eq("both").all():
            raise ValueError(f"Training: missing required {name} applications or labels")
    result["label_available_date"] = result["snapshot_date"].dt.date.map(
        lambda value: label_available_date(value, project["target"]["reporting_lag_months"])
        if pd.notna(value) else None)
    return result.drop(columns="_merge")


def validate_fit_eligibility(frame: pd.DataFrame, fit_as_of: str) -> pd.DataFrame:
    cutoff = month(fit_as_of)
    if cutoff < date(2024, 7, 1):
        raise ValueError("Final Development refit cannot precede 2024-07-01")
    development = frame[frame["split"].eq("development")].copy()
    if development.empty or development["label"].isna().any():
        raise ValueError("Final fit: missing Development labels")
    if not development["label_available_date"].le(cutoff).all():
        raise ValueError("Final fit: Development label not available at fit as-of")
    return development


def validate_model_bundle(bundle: dict) -> None:
    required = {"model_version", "model_name", "model", "feature_columns",
                "numeric_columns", "categorical_columns", "positive_class"}
    if required - set(bundle):
        raise ValueError(f"Model bundle missing {sorted(required - set(bundle))}")
    if bundle["feature_columns"] != bundle["numeric_columns"] + bundle["categorical_columns"]:
        raise ValueError("Model bundle feature order mismatch")
    classes = list(bundle["model"].classes_)
    if classes != [0, 1] or bundle["positive_class"] != 1:
        raise ValueError(f"Model bundle class order invalid: {classes}")
    names = list(bundle["model"].feature_names_)
    if names != bundle["feature_columns"]:
        raise ValueError("Model bundle estimator feature names mismatch")


def validate_predictions(predictions: pd.DataFrame, features: pd.DataFrame,
                         model_version: str) -> dict:
    keys = ["Customer_ID", "application_date"]
    if predictions.duplicated(keys + ["model_version"]).any():
        raise ValueError("Predictions: duplicate canonical keys")
    if not predictions["model_version"].eq(model_version).all():
        raise ValueError("Predictions: model version mismatch")
    values = predictions["p_default"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions: invalid probabilities")
    expected = features[keys].sort_values(keys).reset_index(drop=True)
    actual = predictions[keys].sort_values(keys).reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError("Predictions: keys differ from feature input")
    return {"rows": len(predictions)}
