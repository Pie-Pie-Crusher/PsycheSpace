#!/usr/bin/env python3
"""
preprocess_mental_health.py

Usage:
    python preprocess_mental_health.py --input raw.csv --output processed.csv --outlier-report outliers.csv

What it does:
- Loads a CSV containing columns like country, age, gender, exercise level, diet type,
  sleep hours, stress level, work hours per week, screen time per day, happiness score, etc.
- Handles missing values (numeric and categorical).
- Encodes non-numerical features (one-hot and label encoding where appropriate).
- Flags obvious outliers using IQR and z-score methods. Produces boolean flags per feature and a combined flag.
- Normalizes numeric columns using RobustScaler (robust to outliers) and also provides an option for MinMax/Standard scaling.
- Saves processed dataset and an outlier report.

Author: ChatGPT (GPT-5 Thinking mini)
"""

import argparse
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------
# Configuration / helpers
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def guess_column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Make a best-effort guess of which columns are:
      - numeric_features: continuous/numeric (age, sleep hours, work hours, screen time, happiness, stress, etc.)
      - categorical_features: nominal text features (country, diet type, gender)
      - ordinal_features: if present (e.g., exercise level as 'low','medium','high' etc)
    This helper is conservative — it inspects dtype and a few column name heuristics.
    """
    numeric_features = []
    categorical_features = []
    ordinal_candidates = []

    for col in df.columns:
        lc = col.lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
            continue
        # name-based heuristics
        if any(k in lc for k in ("country", "diet", "gender", "sex", "city", "region", "race")):
            categorical_features.append(col)
        elif any(k in lc for k in ("exercise", "stress", "happiness", "level", "rating", "satisfaction")):
            # could be ordinal or numeric encoded as string; leave to user to confirm if needed
            # If values look numeric-like, treat as numeric:
            sample = df[col].dropna().astype(str).head(20)
            numeric_like = sample.str.match(r"^\d+(\.\d+)?$").all() if len(sample) > 0 else False
            if numeric_like:
                numeric_features.append(col)
            else:
                # treat as ordinal candidate
                ordinal_candidates.append(col)
        else:
            # fallback: if unique values small relative to nrows, treat as categorical
            nunique = df[col].nunique(dropna=True)
            if nunique <= 0.2 * len(df):  # heuristic: many repeated values
                categorical_features.append(col)
            else:
                # try numeric conversion
                try:
                    pd.to_numeric(df[col].dropna().sample(min(20, df[col].dropna().shape[0])))
                    # if conversion works for sample, assume numeric
                    numeric_features.append(col)
                except Exception:
                    categorical_features.append(col)

    return {
        "numeric": sorted(list(dict.fromkeys(numeric_features))),
        "categorical": sorted(list(dict.fromkeys(categorical_features))),
        "ordinal_candidates": sorted(list(dict.fromkeys(ordinal_candidates))),
    }


def flag_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean Series where True = outlier based on IQR method."""
    if series.dropna().empty:
        return pd.Series(False, index=series.index)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def flag_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean Series where True = outlier based on absolute z-score > threshold."""
    # zscore requires numeric without NaN; compute robustly
    if series.dropna().empty:
        return pd.Series(False, index=series.index)
    z = np.abs(stats.zscore(series, nan_policy="omit"))
    # stats.zscore returns array aligned to non-nulls; reindex carefully
    z_series = pd.Series(z, index=series.dropna().index)
    flags = pd.Series(False, index=series.index)
    flags.loc[z_series.index] = z_series > threshold
    return flags


def create_outlier_flags(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Create outlier flags for numeric columns using both IQR and z-score, plus a combined flag.
    Returns a DataFrame with flag columns appended (boolean).
    """
    flags = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            series = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        flags[f"{col}__outlier_iqr"] = flag_outliers_iqr(series)
        flags[f"{col}__outlier_zscore"] = flag_outliers_zscore(series)
        # Combined rule: flagged if either method says outlier OR if value outside reasonable absolute bounds (domain rules)
        # Domain checks for common-sense columns (age, sleep, work hours, screen time, happiness)
        combined = flags[f"{col}__outlier_iqr"] | flags[f"{col}__outlier_zscore"]
        lc = col.lower()
        # reasonable absolute thresholds (obvious impossible/implausible values)
        if "age" in lc:
            combined |= (series < 5) | (series > 120)  # human-lifespan bounds
        if "sleep" in lc or "sleep_hours" in lc or "sleep" in lc:
            combined |= (series < 0) | (series > 24)
        if "work" in lc or "work_hours" in lc:
            combined |= (series < 0) | (series > 120)
        if "screen" in lc or "screen_time" in lc:
            combined |= (series < 0) | (series > 24 * 7)  # if reported in hours/day, >168 is impossible
        if "happiness" in lc or "stress" in lc or "score" in lc:
            # many scales are 0-10 or 1-10; flag values outside -50..150 as absurd
            combined |= (series < -50) | (series > 150)
        flags[f"{col}__outlier_combined"] = combined.fillna(False)
    # add a combined row-level flag
    flags["any_outlier"] = flags.filter(like="__outlier_combined").any(axis=1)
    return flags


def summarize_outliers(flags_df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary: count of outliers per column and indices."""
    outlier_cols = [c for c in flags_df.columns if c.endswith("__outlier_combined")]
    summary = []
    for col in outlier_cols:
        count = flags_df[col].sum()
        indices = flags_df.index[flags_df[col]].tolist()
        summary.append({"feature": col.replace("__outlier_combined", ""), "outlier_count": int(count), "outlier_indices": indices})
    return pd.DataFrame(summary)


# -----------------------
# Main processing pipeline
# -----------------------
def build_preprocessing_pipeline(numeric_cols: List[str], categorical_cols: List[str], ordinal_cols: List[str] = None):
    """
    Build sklearn ColumnTransformer pipeline:
    - Numeric: impute (median) then RobustScaler
    - Categorical: impute (most_frequent) then OneHotEncoder (drop='first')
    - Ordinal: if provided — do OrdinalEncoder with unknown handling
    Returns (preprocessor, output_feature_names) where preprocessor is a ColumnTransformer.
    """
    if ordinal_cols is None:
        ordinal_cols = []

    # Numeric pipeline: median imputation then RobustScaler (less sensitive to remaining outliers)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),  # robust scaling is often preferred with outliers
        ]
    )

    # Categorical pipeline: impute with most frequent then one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore", drop="first")),
        ]
    )

    # Ordinal pipeline: if present, we treat as ordered categories; user may need to supply order mapping.
    # Here we simply label-encode (internal ordinal) but DO NOT assume order directionality.
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if ordinal_cols:
        transformers.append(("ord", ordinal_pipeline, ordinal_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

    return preprocessor


def get_feature_names_from_column_transformer(ct: ColumnTransformer, input_df: pd.DataFrame) -> List[str]:
    """
    Try to assemble output feature names after ColumnTransformer.transform.
    Note: OneHotEncoder created feature names are extracted from the OneHotEncoder instance.
    """
    feature_names = []
    for name, transformer, columns in ct.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if transformer == "drop":
            continue
        # sklearn pipelines: transformer may be a Pipeline
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            ohe = transformer.named_steps["onehot"]
            # ohe.get_feature_names_out needs the original column names
            names = list(ohe.get_feature_names_out(columns))
            feature_names.extend(names)
        elif hasattr(transformer, "named_steps") and "scaler" in transformer.named_steps:
            feature_names.extend(columns)
        elif hasattr(transformer, "get_feature_names_out"):
            try:
                names = list(transformer.get_feature_names_out(columns))
                feature_names.extend(names)
            except Exception:
                feature_names.extend(columns)
        else:
            feature_names.extend(columns)
    return feature_names


def preprocess_dataframe(df: pd.DataFrame, output_scaled: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main wrapper: detect column roles, create outlier flags, run preprocessing, return (processed_df, outlier_flags_df)
    - output_scaled True returns scaled numeric columns (RobustScaler).
    """
    logging.info("Guessing column roles (numeric/categorical/ordinal candidates)")
    roles = guess_column_roles(df)
    numeric_cols = roles["numeric"]
    categorical_cols = roles["categorical"]
    ordinal_candidates = roles["ordinal_candidates"]

    logging.info(f"Numeric columns detected: {numeric_cols}")
    logging.info(f"Categorical columns detected: {categorical_cols}")
    logging.info(f"Ordinal candidate columns: {ordinal_candidates}")

    # 1) Create outlier flags BEFORE imputation/scaling (operates on original values)
    logging.info("Creating outlier flags for numeric columns...")
    outlier_flags = create_outlier_flags(df, numeric_cols)

    # 2) Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols, ordinal_cols=ordinal_candidates)

    # Fit the preprocessor and transform
    logging.info("Fitting preprocessing pipeline and transforming data...")

    # Work on a copy so we don't lose original raw columns
    working_df = df.copy()
    # For columns that are numeric but provided as strings with commas, normalize decimal marks (common in international datasets)
    for c in numeric_cols:
        if not pd.api.types.is_numeric_dtype(working_df[c]):
            # replace comma decimal separators (e.g., "7,5") with dot and try convert
            working_df[c] = working_df[c].astype(str).str.replace(",", ".").replace({"nan": np.nan})
            working_df[c] = pd.to_numeric(working_df[c], errors="coerce")

    # Fit
    preprocessor.fit(working_df)

    transformed = preprocessor.transform(working_df)
    # get output columns names
    try:
        feature_names = get_feature_names_from_column_transformer(preprocessor, working_df)
    except Exception:
        # fallback: combine names
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            # one-hot will expand; we cannot easily name them here, so keep original names (user can inspect)
            feature_names.extend(categorical_cols)
        if ordinal_candidates:
            feature_names.extend(ordinal_candidates)

    processed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)

    # Append original non-processed columns that were not part of pipelines (if any) to keep context
    leftover_cols = [c for c in df.columns if c not in numeric_cols + categorical_cols + ordinal_candidates]
    if leftover_cols:
        logging.info(f"Appending leftover columns without transformation: {leftover_cols}")
        processed_df = pd.concat([processed_df, df[leftover_cols].reset_index(drop=True)], axis=1)

    # 3) Optionally apply another normalization method if user prefers — already applied RobustScaler to numeric.
    # If you want MinMax or Standard scaling instead, you can swap the scaler in build_preprocessing_pipeline
    # For clarity, we keep the scaled numeric columns (RobustScaled) in processed_df.

    # 4) Combine processed_df with outlier flags
    combined = pd.concat([processed_df.reset_index(drop=True), outlier_flags.reset_index(drop=True)], axis=1)

    # 5) Provide a simple warning/summary for review
    num_outliers_total = outlier_flags["any_outlier"].sum()
    logging.info(f"Total rows with any obvious outlier: {int(num_outliers_total)} out of {len(df)}")

    return combined, outlier_flags


# -----------------------
# CLI / Entrypoint
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess mental-health dataset (missing values, encoding, outlier flags, normalization).")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to save processed CSV file")
    parser.add_argument("--outlier-report", "-r", required=False, default="outliers_report.csv", help="Path to save outlier report CSV")
    parser.add_argument("--no-save", action="store_true", help="If set, don't write output file (useful for testing)")
    args = parser.parse_args()

    logging.info(f"Loading input data from {args.input} ...")
    df = pd.read_csv(args.input)

    # Basic sanity info
    logging.info(f"Input shape: {df.shape}")
    logging.info("Columns: " + ", ".join(df.columns.tolist()))

    processed_df, flags_df = preprocess_dataframe(df)

    # Save processed dataset (unless no-save)
    if not args.no_save:
        logging.info(f"Saving processed dataset to {args.output} ...")
        processed_df.to_csv(args.output, index=False)
        logging.info(f"Saving outlier report to {args.outlier_report} ...")
        # Summarize outliers: counts and indices
        summary = summarize_outliers(flags_df)
        summary.to_csv(args.outlier_report, index=False)

    logging.info("Done. Inspect the processed file and the outlier report for manual review before modelling.")
    logging.info("Notes: - Automated outlier detection is a fast filter but not a substitute for domain review.")
    logging.info("       - If your dataset uses different naming conventions (e.g., 'SleepHours' vs 'sleep_hours'), consider renaming columns for clarity.")
    logging.info("       - If you want different scaling (MinMax/Standard) edit build_preprocessing_pipeline() accordingly.")


if __name__ == "__main__":
    main()
