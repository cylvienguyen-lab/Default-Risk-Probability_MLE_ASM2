"""Frozen final feature selection and CatBoost input preparation."""

import numpy as np
import pandas as pd


def selected_features(feature_config: dict, model_config: dict) -> dict:
    excluded_groups = set(model_config["excluded_groups"])
    excluded_columns = set(model_config["excluded_columns"])
    groups = [value for name, value in feature_config["predictors"].items()
              if name not in excluded_groups]
    numeric = [column for group in groups for column in group["numerical"]["non_nullable"]]
    numeric += [column for group in groups for column in group["numerical"]["nullable"]]
    numeric = [column for column in numeric if column not in excluded_columns]
    categorical = [column for group in groups for column in group["categorical"]
                   if column not in excluded_columns]
    if len(numeric) != 29 or len(categorical) != 4 or len(set(numeric + categorical)) != 33:
        raise ValueError("Frozen CatBoost recipe must yield 29 numeric and 4 categorical fields")
    nullable = [column for group in groups for column in group["numerical"]["nullable"]
                if column in numeric]
    return {"feature_columns": numeric + categorical, "numeric_columns": numeric,
            "categorical_columns": categorical, "nullable_numeric_columns": nullable}


def prepare_model_input(frame: pd.DataFrame, feature_spec: dict) -> pd.DataFrame:
    columns = feature_spec["feature_columns"]
    numeric = feature_spec["numeric_columns"]
    categorical = feature_spec["categorical_columns"]
    if columns != numeric + categorical or len(columns) != len(set(columns)):
        raise ValueError("Invalid ordered model feature specification")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Model input missing columns: {sorted(missing)}")
    result = frame.loc[:, columns].copy()
    nullable = set(feature_spec.get("nullable_numeric_columns", []))
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"Model numeric column has incompatible type: {column}")
        values = result[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (column not in nullable and np.isnan(values).any()):
            raise ValueError(f"Model numeric column has invalid values: {column}")
    for column in categorical:
        if result[column].isna().any():
            raise ValueError(f"Model categorical column has null values: {column}")
        result[column] = result[column].astype(str)
    return result
