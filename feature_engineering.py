"""
feature_engineering module.

This module contains feature engineering utilities for processing
time-series and sequential process data originating from an industrial
manufacturing process. It provides functions for computing relative
features, temporal derivatives, rolling statistics, cumulative metrics,
and vector magnitudes, and is designed to operate both on pandas
DataFrames and Snowflake Snowpark DataFrames.
"""

import logging
import os
import __main__

import pandas as pd
import numpy as np
from snowflake.snowpark.types import StructType, StructField, StringType, FloatType
from snowflake.snowpark import Session, DataFrame as SnowparkDataFrame

logger = logging.getLogger(__name__)


def add_relative_features(
    df: pd.DataFrame,
    start_index: int = 75,
    end_index: int = 125,
    feature_list: list[str] = None,
) -> pd.DataFrame:
    """
    Add relative features normalized by per-part median setpoints.

    For each feature in `feature_list`, this function computes a median
    setpoint value within a specified subpart index range for each part.
    The original feature values are then divided by the corresponding
    setpoint, producing relative (dimensionless) features.

    Args:
        df (pd.DataFrame): Input DataFrame containing process data.
        start_index (int): Lower bound of subpart_index range used
            to compute median setpoints.
        end_index (int): Upper bound of subpart_index range used
            to compute median setpoints.
        feature_list (list[str], optional): List of feature column names
            for which relative features should be computed. If None,
            defaults to ["first_param", "second_param", "third_param"].

    Returns:
        pd.DataFrame: DataFrame with additional columns named
        `relative_<feature>` for each feature in `feature_list`.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning as is.")
        return df

    if feature_list is None:
        feature_list = ["first_param", "second_param", "third_param"]

    required_cols = ["subpart_index", "part_number"] + feature_list
    if not set(required_cols).issubset(df.columns):
        missing_cols = set(required_cols) - set(df.columns)
        logger.error("DataFrame do not contain required columns: %s", missing_cols)
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    logger.info("Adding relative features for: %s ...", feature_list)

    mask = (df["subpart_index"] >= start_index) & (df["subpart_index"] <= end_index)
    middle_range_df = df.loc[mask]

    if middle_range_df.empty:
        logger.warning("""No data found in the middle range. Cannot calculate setpoints.
                        Returning NaN as relative features in DataFrame""")

        for feature in feature_list:
            df[f"relative_{feature}"] = np.nan
        return df

    median_setpoints = middle_range_df.groupby("part_number")[feature_list].median()

    setpoint_cols_map = {feature: f"{feature}_setpoint" for feature in feature_list}
    median_setpoints = median_setpoints.rename(columns=setpoint_cols_map)

    enhanced_df = df.merge(median_setpoints, left_on="part_number", right_index=True, how="left")

    for feature, setpoint_col in setpoint_cols_map.items():
        new_feature_name = f"relative_{feature}"

        denominator = enhanced_df[setpoint_col] + 1e-9

        enhanced_df[new_feature_name] = enhanced_df[feature] / denominator

    columns_to_drop = list(setpoint_cols_map.values())
    enhanced_df = enhanced_df.drop(columns = columns_to_drop)

    return enhanced_df


def compute_derivative(sequence, order=1):
    """
    Compute the numerical derivative of a sequence.

    Uses numpy.gradient to estimate the derivative. Higher-order
    derivatives are computed iteratively.

    Args:
        sequence (array-like): Input numeric sequence.
        order (int): Order of the derivative to compute.

    Returns:
        pd.Series: The computed derivative sequence.
    """
    for _ in range(order):
        sequence = np.gradient(sequence, axis=0)
    return pd.Series(sequence)


def compute_moving_average(sequence, window_size=3):
    """
    Compute a centered moving average over a sequence.

    Args:
        sequence (array-like): Input numeric sequence.
        window_size (int): Size of the rolling window.

    Returns:
        pd.Series: Moving average values.
    """
    return pd.Series(sequence).rolling(window=window_size, min_periods=1, center=True).mean()


def compute_moving_std(sequence, window_size=3):
    """
    Compute a centered moving standard deviation over a sequence.

    Args:
        sequence (array-like): Input numeric sequence.
        window_size (int): Size of the rolling window.

    Returns:
        pd.Series: Moving standard deviation values.
    """
    return pd.Series(sequence).rolling(window=window_size, min_periods=1, center=True).std()


def compute_differences(sequence):
    """
    Compute differences between neighboring elements of a sequence.

    The first element is preserved by prepending the initial value.

    Args:
        sequence (array-like or pd.Series): Input numeric sequence.

    Returns:
        pd.Series: Sequence of first-order differences.
    """
    first_value = sequence.iloc[0] if isinstance(sequence, pd.Series) else sequence[0]
    diffs = np.diff(sequence, prepend=first_value)
    return pd.Series(diffs)


def compute_cumsum_differences(sequence):
    """
    Compute the cumulative sum of neighboring differences.

    Args:
        sequence (array-like or pd.Series): Input numeric sequence.

    Returns:
        pd.Series: Cumulative sum of differences.
    """
    return compute_differences(sequence).cumsum()


def extract_features(df, feature_name, group_by_columns):
    """
    Extract temporal and statistical features for a single signal.

    Features are computed independently for each group defined by
    `group_by_columns`.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_name (str): Name of the feature column to process.
        group_by_columns (list[str]): Columns used to define groups.

    Returns:
        dict[str, pd.Series]: Dictionary mapping feature names to
        computed feature Series.
    """
    grouped = df.groupby(group_by_columns)[feature_name]

    # Dla feature "machine_position" chcemy obliczyć cumsum z abs(diff)
    if feature_name == "machine_position":
        def diff_cumsum_func(x):
            return compute_differences(x).abs().cumsum()
    else:
        diff_cumsum_func = compute_cumsum_differences

    return {
        'velocity': grouped.apply(compute_derivative).explode().astype(float),
        'acceleration': grouped.apply(lambda x: compute_derivative(x, order=2))
                                                        .explode().astype(float),
        'moving_average': grouped.apply(compute_moving_average).explode().astype(float),
        'moving_std': grouped.apply(compute_moving_std).explode().astype(float),
        'neighbor_diff': grouped.apply(compute_differences).explode().astype(float),
        'neighbor_diff_cumsum': grouped.apply(diff_cumsum_func).explode().astype(float),
    }


def compute_vector_magnitude(sequences):
    """
    Compute the Euclidean magnitude of 3D vectors.

    Args:
        sequences (pd.DataFrame): DataFrame containing three columns
            representing vector components.

    Returns:
        pd.Series: Magnitude of the vectors.
    """
    sequences = sequences.astype(float)
    squared_sum = sequences.iloc[:, 0]**2 + sequences.iloc[:, 1]**2 + sequences.iloc[:, 2]**2

    valid_sum = np.where(squared_sum >= 0, squared_sum, np.nan)
    with np.errstate(invalid='ignore'):
        magnitude = np.sqrt(valid_sum)
    return magnitude


def extract_vector_magnitude(df, feature_names, group_by_columns):
    """
    Extract vector magnitude feature for grouped 3D signals.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_names (list[str]): Names of three feature columns
            forming a vector.
        group_by_columns (list[str]): Columns used to define groups.

    Returns:
        dict[str, pd.Series]: Dictionary containing the magnitude feature.
    """
    grouped = df.groupby(group_by_columns)[feature_names]

    return {
        'magnitude': grouped.apply(compute_vector_magnitude).explode().astype(float)
    }


def sp_add_all_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature extraction pipeline to a pandas DataFrame.

    This function is designed to be used inside a Snowpark
    apply_in_pandas call.

    Args:
        pdf (pd.DataFrame): Input pandas DataFrame for a single group.

    Returns:
        pd.DataFrame: DataFrame enriched with engineered features.
    """
    pdf = pdf.sort_values(by='row_index')

    for param in ("first_param", "second_param", "third_param"):
        feats = extract_features(pdf, param, ["part_number", "subpart_no"])
        for feat_name, series in feats.items():
            pdf[f"{param}_{feat_name}"] = series.values

    mag = extract_vector_magnitude(
        pdf,
        ["third_param", "second_param", "first_param"],
        ["part_number", "subpart_no"]
    )
    pdf["magnitude"] = mag["magnitude"].values
    return pdf


def prepare_feature_eng_schema(
    sp: SnowparkDataFrame,
    feaure_eng_schema_key: str
) -> StructType:
    """
    Dynamically prepare a Snowpark schema including engineered features.

    Args:
        sp (SnowparkDataFrame): Input Snowpark DataFrame.
        feaure_eng_schema_key (str): Key used to retrieve the target
            schema definition from project configuration.

    Returns:
        StructType: Final Snowflake schema including new feature columns.
    """
    import project_globals as cfg

    existing_fields = sp.schema.fields
    existing_col_names = {field.name.strip('"') for field in existing_fields}
    logger.info("Found %d existing columns.", len(existing_col_names))

    schema = list(cfg.get_project_variable(feaure_eng_schema_key))

    type_overrides = {
        "row_index": FloatType(),
        "device": StringType(),
        "part_number": StringType(),
        "part_index": FloatType(),
        "subpart_index": FloatType(),
        "subpart_no": FloatType(),
        "machine_position": FloatType(),
        "third_param": FloatType(),
        "second_param": FloatType(),
        "first_param": FloatType(),
        "temp_a": FloatType(),
        "temp_b": FloatType(),
        "ng_valve_no": StringType(),
        "subpart_label": FloatType()
    }

    new_fields = []
    for col_name in schema:
        if col_name not in existing_col_names:
            field_type = type_overrides.get(col_name, FloatType())
            new_fields.append(StructField(f'"{col_name}"', field_type, True))

    logger.info("Identified %d new columns to add to the schema.", len(new_fields))

    final_schema = StructType(existing_fields + new_fields)

    logger.info("StructType feature eng schema created dynamically.")

    return final_schema


def apply_feature_engineering(
    sp: SnowparkDataFrame,
    session: Session
) -> SnowparkDataFrame:
    """
    Apply feature engineering to a Snowpark DataFrame.

    This function orchestrates schema preparation and execution of
    pandas-based feature extraction using Snowpark UDFs.

    Args:
        sp (SnowparkDataFrame): Input Snowpark DataFrame.
        session (Session): Active Snowpark session.

    Returns:
        SnowparkDataFrame: Feature-enriched Snowpark DataFrame.

    Raises:
        Exception: Re-raises any exception encountered during processing.
    """
    import project_globals as cfg

    execution_mode = cfg.EXECUTION_MODE
    logger.info("--- Starting feature eng process in mode: %s ---", execution_mode)

    session.add_packages("snowflake-snowpark-python", "pandas", "numpy")
    session.add_import(__file__)

    schema = prepare_feature_eng_schema(sp, cfg.FEATUE_ENG_SCHEMA_KEY)

    if sp.count() == 0:
        logger.warning("Input Snowpark DataFrame is empty. Returning empty Snowpark DataFrame")
        return session.create_dataframe([], schema=schema)

    group_cols = ['"part_number"', '"subpart_no"']

    try:
        num_subparts = sp.select(group_cols).distinct().count()
        logger.info(
            "Applying feature engineering UDF to %d subparts of Snowpark DataFrame...",
            num_subparts
        )

        enhanced_sp = sp.group_by(group_cols).apply_in_pandas(
            sp_add_all_features,
            output_schema=schema
        )

        logger.info("Succesfully processed and created features, rows: %d", enhanced_sp.count())
        logger.info("--- Feature eng process Completed Successfully: %s ---\n", execution_mode)

        return enhanced_sp

    except Exception as e:
        logger.error("An error occurred during data feature eng process: %s", e, exc_info=True)
        raise
