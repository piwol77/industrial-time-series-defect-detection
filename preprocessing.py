"""
feature_engineering module.

This module provides feature engineering and preprocessing utilities for
process data originating from an industrial production process.

The functions in this module operate on sequential, group-based process data
and support tasks such as:
- normalization of subpart identifiers,
- filtering incomplete or abnormal production runs,
- generation of sequential indices for time-series measurements,
- alignment and padding of variable-length process sequences,
- preparation of datasets for machine learning workflows,
- group-aware splitting into train, validation, and test sets,
- mapping process quality labels to binary values.

"""

import logging
import warnings
import os

import project_globals as cfg
import snowflake_data_ingestion as sdi

import pandas as pd
from sklearn.model_selection import train_test_split

# Ignoruj specyficzne ostrzeżenie DeprecationWarning z pkg_resources
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

logger = logging.getLogger(__name__)

def _adjust_number(num: float) -> float:
    """
    Normalize a raw numeric subpart identifier.

    This helper function adjusts subpart number values to a consistent numeric
    representation. Fractional values in the range [0.0, 0.6] are scaled,
    indicating that subpart identifiers may be encoded differently in the
    upstream data source.

    Args:
        num (float): Raw subpart number value.

    Returns:
        float: Normalized subpart number.
    """
    if 0.0 <= num <= 6.0:
        if 0.0 <= num <= 0.6:
            return float(num * 10)
        else:
            return float(num)
    else:
        return float(num)


def normalize_subpart_no_and_exclude_0_subparts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize subpart numbers and exclude records with subpart number equal to zero.

    Seat number `0` is treated as invalid or irrelevant for downstream
    processing, suggesting that only subparts in the range [1–6] participate
    in the production process.

    Args:
        df (pd.DataFrame): Input DataFrame containing a `subpart_no` column.

    Returns:
        pd.DataFrame: DataFrame with normalized subpart numbers and subpart 0 removed.
    """
    logger.info("Normalizing subpart_no and filtering for [1-6] subparts...")

    df["subpart_no"] = df["subpart_no"].astype(float).apply(_adjust_number)
    df_excluded_0 = df[df["subpart_no"] != 0]

    return df_excluded_0


def filter_by_data_len(df: pd.DataFrame, no_processed_subparts: int = 6) -> pd.DataFrame:
    """
    Filter production runs based on expected sequence length per part.

    This function removes part-level groups whose number of records deviates
    significantly from the expected number of time steps. This typically
    filters out interrupted, duplicated, or incomplete production cycles.

    Args:
        df (pd.DataFrame): Input DataFrame containing a `part_number` column.
        no_processed_subparts (int): Expected number of processed subparts per part.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid production runs.

    Raises:
        KeyError: If the `part_number` column is missing.
    """
    logger.info("Filtering for duplicates and trimmed data...")
    if "part_number" not in df.columns:
        raise KeyError("Required column 'part_number' not found in DataFrame.")

    time_steps_per_subpart = 230 if df["subpart_no"].min() == 0 else 200
    lower_threshold = (no_processed_subparts * time_steps_per_subpart) * 0.9
    upper_threshold = (no_processed_subparts * time_steps_per_subpart) * 1.1

    parts_count = df["part_number"].value_counts()
    valid_parts = parts_count[
        (parts_count > lower_threshold) & (parts_count < upper_threshold)
    ].index

    filtered_df = df[df["part_number"].isin(valid_parts)]
    logger.info("Filtered out %d groups based on length.", len(parts_count) - len(valid_parts))
    return filtered_df


def insert_subpart_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert a sequential index for each subpart within a part.

    The generated `subpart_index` represents the temporal or step-based order
    of measurements for each (`part_number`, `subpart_no`) combination.

    Args:
        df (pd.DataFrame): Input DataFrame containing `part_number` and `subpart_no`.

    Returns:
        pd.DataFrame: DataFrame with an added `subpart_index` column.

    Raises:
        KeyError: If required columns are missing.
    """
    logger.info("Inserting subpart_index column...")
    df_copy = df.copy()

    required_columns = ["part_number", "subpart_no"]
    for col_name in required_columns:
        if col_name not in df_copy.columns:
            raise KeyError(f"Required column '{col_name}' not found in the DataFrame.")

    df_copy.insert(
        2, "subpart_index", df_copy.groupby(["part_number", "subpart_no"]).cumcount().astype(float) + 1
    )

    return df_copy.reset_index(drop=True)


def concat_dfs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate two DataFrames row-wise.

    This function is typically used to combine datasets representing
    different quality classes (e.g. OK and NG) for training purposes.

    Args:
        df_a (pd.DataFrame): First DataFrame.
        df_b (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    return pd.concat([df_a, df_b], axis=0)


# BEZ USUWANIA NIEPRAWIDLOWYCH PRZEBIEGOW NA PODSTAWIE MEDIAN POWDER_FLOW


def align_seqs_len(
    df: pd.DataFrame,
    target_len: int,
    group_cols: list[str] = None,
    subpart_index_col: str = "subpart_index",
) -> pd.DataFrame:
    """
    Pad all sequences within each group to a fixed target length.

    If a sequence is shorter than the target length, additional rows are
    appended and filled using the last observed values within the group,
    effectively carrying forward the final known process state.

    Args:
        df (pd.DataFrame): Input DataFrame containing sequential data.
        target_len (int): Target sequence length.
        group_cols (list[str], optional): Columns defining a sequence group.
        subpart_index_col (str): Name of the sequence index column.

    Returns:
        pd.DataFrame: DataFrame with padded and aligned sequences.
    """
    logger.info("Padding all sequences to target length %d...", target_len)
    if group_cols is None:
        group_cols = ["part_number", "subpart_no"]

    def _extend_group(group: pd.Series) -> pd.Series:
        """A dummy description."""
        current_length = len(group)
        max_subpart_index = int(group[subpart_index_col].max())
        if current_length < target_len:
            additional_rows = pd.DataFrame(
                {
                    subpart_index_col: range(max_subpart_index + 1, target_len + 1),
                }
            )
            for col in group_cols + list(group.columns):
                if col not in [subpart_index_col]:
                    additional_rows[col] = group[col].iloc[-1]
            group = pd.concat([group, additional_rows], ignore_index=True)
        return group

    orig_types = df.dtypes

    extended_df = df.groupby(group_cols).apply(_extend_group).reset_index(drop=True)

    for col in extended_df.columns:
        extended_df[col] = extended_df[col].astype(orig_types[col])


    return extended_df


def align_seqs_len_optimized(
    df: pd.DataFrame,
    target_len: int,
    group_cols: list[str] = None,
    subpart_index_col: str = "subpart_index",
) -> pd.DataFrame:
    """
    Align and pad sequences to a fixed length using an optimized approach.

    This implementation uses MultiIndex reindexing and forward-filling to
    efficiently extend all sequences to the target length.

    Args:
        df (pd.DataFrame): Input DataFrame containing sequential data.
        target_len (int): Target sequence length.
        group_cols (list[str], optional): Columns defining a sequence group.
        subpart_index_col (str): Name of the sequence index column.

    Returns:
        pd.DataFrame: DataFrame with aligned and padded sequences.
    """

    if group_cols is None:
        group_cols = ["part_number", "subpart_no"]

    orig_types = df.dtypes

    unique_groups = df[group_cols].drop_duplicates()

    new_index = pd.MultiIndex.from_product(
        [unique_groups[col] for col in group_cols] + [range(1, target_len + 1)],
        names=group_cols + [subpart_index_col],
    )

    extended_df = df.set_index(group_cols + [subpart_index_col]).reindex(new_index)

    extended_df = extended_df.groupby(level=group_cols).ffill()

    extended_df.reset_index(inplace=True)

    for col, dtype in orig_types.items():
        if col in extended_df.columns and pd.api.types.is_numeric_dtype(dtype):
            extended_df[col] = extended_df[col].fillna(0).astype(dtype)
        elif col in extended_df.columns:
            extended_df[col] = extended_df[col].astype(dtype)


    return extended_df


def insert_part_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert a sequential index for each production part.

    The generated `part_index` represents the order of records within each
    `part_number` group. This index can be interpreted as a time-step or
    processing-step counter for an entire production part.

    Args:
        df (pd.DataFrame): Input DataFrame containing a `part_number` column.

    Returns:
        pd.DataFrame: DataFrame with an added `part_index` column.

    Raises:
        KeyError: If the `part_number` column is missing.
    """
    logger.info("Inserting part_index column...")
    df_copy = df.copy()

    part_col = "part_number"
    if part_col not in df_copy.columns:
        raise KeyError(f"Required column '{part_col}' not found in the DataFrame.")

    df_copy.insert(
        2, "part_index", df_copy.groupby([part_col]).cumcount().astype(float) + 1
    )

    return df_copy.reset_index(drop=True)


def exclude_train_lables_ok_within_ng(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude intermediate quality labels from training data.

    This function removes records labeled as `OK_within_NG`, which appear to
    represent ambiguous or transitional quality states that should not be
    used during model training.

    Args:
        df (pd.DataFrame): Input DataFrame containing a `subpart_label` column.

    Returns:
        pd.DataFrame: Filtered DataFrame without `OK_within_NG` labels.
    """
    logger.info("Excluding 'OK_within_NG' labels for training mode...")

    filtered_df = df[df["subpart_label"] != "OK_within_NG"]

    return filtered_df


def get_max_from_column(
    df: pd.DataFrame,
    column_name: str
) -> pd.DataFrame:
    """
    Retrieve the maximum value from a specified column.

    This function is commonly used to determine the maximum sequence length
    (e.g. maximum `subpart_index`) in the dataset, which can later be reused
    as a global reference value.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to evaluate.

    Returns:
        int: Maximum value found in the column, or 0 if the DataFrame is empty.

    Raises:
        KeyError: If the specified column does not exist.
    """
    if df.empty:
        logger.warning("Input df for column '%s' is empty. Returning 0.", column_name)
        return 0

    if column_name not in df.columns:
        logger.error("Column '%s' not found in df. Aborting.", column_name)
        return KeyError(f"Required column {column_name} not found")

    return int(df[column_name].max())


def split_data_by_group(
    df: pd.DataFrame,
    label_col: str = None,
    group_cols: list[str] = None,
    train_size: float = 0.75,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets using group-based sampling.

    The split is performed at the group level (e.g. by `part_number` and
    `subpart_no`) to ensure that all records belonging to the same production
    unit are assigned to the same dataset. Stratification is applied based
    on the label column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_col (str, optional): Name of the label column.
        group_cols (list[str], optional): Columns defining a group.
        train_size (float): Proportion of groups assigned to training.
        val_size (float): Proportion of groups assigned to validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, validation, and test DataFrames.

    Raises:
        ValueError: If train and validation sizes exceed 1.0.
    """
    if label_col is None:
        label_col = 'subpart_label'
    if group_cols is None:
        group_cols = ['part_number', 'subpart_no']

    test_size = 1.0 - train_size - val_size
    if test_size < 0:
        raise ValueError("Sum of Train_size and val_size cannot be higher than 1.0")

    logger.info("Splitting data into train/val/test sets by group...")
    unique_subparts = df[group_cols + [label_col]].drop_duplicates()

    train_groups, temp_subparts = train_test_split(
        unique_subparts,
        train_size=train_size,
        random_state=random_state,
        stratify=unique_subparts[label_col],
    )

    val_proportion_in_temp = val_size / (val_size + test_size)

    val_subparts, test_subparts = train_test_split(
        temp_subparts,
        train_size=val_proportion_in_temp,
        random_state=random_state,
        stratify=temp_subparts[label_col],
    )

    def _filter_by_groups(original_df, groups_to_keep):
        return original_df.merge(groups_to_keep[group_cols], on=group_cols, how="inner")

    train_df = _filter_by_groups(df, train_groups)
    val_df = _filter_by_groups(df, val_subparts)
    test_df = _filter_by_groups(df, test_subparts)

    logger.info(
        "Split completed. Train shape: %s, Val shape: %s, Test shape: %s",
        train_df.shape, val_df.shape, test_df.shape
    )

    return train_df, val_df, test_df


def map_subpart_labels_to_binary(
    df: pd.DataFrame,
    label_col: str = None,
) -> pd.DataFrame:
    """
    Map categorical subpart quality labels to binary numeric values.

    The mapping is defined as:
    - `NG` → 1
    - `OK` → 0

    This prepares the label column for binary classification models.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_col (str, optional): Name of the label column.

    Returns:
        pd.DataFrame: DataFrame with binary-encoded labels.

    Raises:
        KeyError: If the label column is missing.
    """
    logger.info("Mapping subpart labels to binary values...")

    if label_col is None:
        label_col = 'subpart_label'
    if label_col not in df.columns:
        logger.error("Column '%s' not found in df. Aborting.", label_col)
        return KeyError(f"Required column {label_col} not found")

    df_copy = df.copy()

    label_map = {"NG": 1, "OK": 0}
    df_copy[label_col] = df_copy[label_col].map(label_map)

    df_copy[label_col] = df_copy[label_col].astype(float)

    return df_copy


def process_data(
    first_df: pd.DataFrame,
    second_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline for process data.

    This function orchestrates all preprocessing steps required to transform
    raw production process data into a model-ready format. Depending on the
    execution mode, it handles both training and inference workflows.

    The pipeline includes:
    - normalization of subpart identifiers,
    - filtering incomplete production runs,
    - generation and alignment of sequential indices,
    - optional exclusion and mapping of training labels.

    Args:
        first_df (pd.DataFrame): Primary input DataFrame.
        second_df (pd.DataFrame, optional): Secondary DataFrame used in training
            mode (e.g. combining OK and NG samples).

    Returns:
        pd.DataFrame: Fully processed DataFrame ready for modeling.

    Raises:
        Exception: Propagates any exception encountered during processing.
    """
    execution_mode = cfg.EXECUTION_MODE
    logger.info("--- Starting Preprocessing data in mode: %s ---", execution_mode)

    if first_df.empty or (second_df is not None and second_df.empty):
        logger.warning("Input DataFrame is empty, skipping processing.")
        return pd.DataFrame()

    try:
        if second_df is not None:
            logger.info("Combining OK and NG files for training mode...")

        initial_df = (
            concat_dfs(first_df, second_df) if second_df is not None else first_df
        )
        logger.info("Initial pandas DataFrame loaded. Total records: %d", len(initial_df))

        partially_processed_df = (
            initial_df.pipe(normalize_subpart_no_and_exclude_0_subparts)
            .pipe(filter_by_data_len)
            .pipe(insert_subpart_index)
        )

        if "train" in execution_mode:
            target_length = get_max_from_column(partially_processed_df, "subpart_index")
            cfg.modify_project_variable(cfg.MAX_TRAIN_SEQ_LENGTH_VAR_KEY, target_length)
        else:
            target_length = cfg.get_project_variable(cfg.MAX_TRAIN_SEQ_LENGTH_VAR_KEY)

        target_length = int(target_length)

        processed_df = (
            partially_processed_df.pipe(align_seqs_len, target_len=target_length)
            .pipe(insert_part_index)
        )

        if "train" in execution_mode:
            processed_df = exclude_train_lables_ok_within_ng(processed_df)
            processed_df = map_subpart_labels_to_binary(processed_df)

        logger.info("Processed pandas DataFrame total records: %d", len(processed_df))

        logger.info("--- Preprocessing Completed Successfully: %s ---\n", execution_mode)

        return processed_df

    except Exception as e:
        logger.error("An error occurred during data pre-processing: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    print(f"--- Running local test of {os.path.basename(__file__)} module ---")
    from dataiku.snowpark import DkuSnowpark

    session = DkuSnowpark().get_session(
        connection_name=cfg.get_project_variable(cfg.SNOWFLAKE_CONNECTION_VAR_KEY)
    )

    try:
        TEST_TIMESTAMP = 1754069399048
        SHOULD_UPDATE_GLOBALS = False

        testing_df = sdi.get_and_prepare_data_from_snowflake(
            session,
            test_last_timestamp_ms=TEST_TIMESTAMP,
            test_update_globals=SHOULD_UPDATE_GLOBALS,
        )

        final_df = process_data(testing_df)

        print("\n--- Test Run Results ---")
        if not final_df.empty:
            print(f"Final processed Pandas DataFrame has {len(final_df)} records.")
            print(final_df.info())
            print(final_df.head(402))
        else:
            print("No data retrieved. Final DataFrame is empty.")

    except (ValueError, KeyError, RuntimeError) as e:
        print(f"TEST FAILED during data processing. Reason: {e}")
