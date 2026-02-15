"""
snowflake_data_ingestion module

This module is responsible for ingesting raw production process data from
Snowflake and preparing it for downstream feature engineering and modeling.

It supports incremental data loading based on timestamps, extraction and
flattening of JSON-encoded process signals, and conversion to a Pandas
DataFrame suitable for machine learning pipelines.
"""

import os
import logging
import pandas as pd
import __main__
from snowflake.snowpark import dataframe as SnowparkDataFrame, Session
from snowflake.snowpark.functions import col, lit, parse_json
from snowflake.snowpark.types import StringType, FloatType

import project_globals as cfg
import utils

logging.getLogger('snowflake').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def _return_sp_newer_than_timestamp_files(
    sp_df: SnowparkDataFrame,
    last_timestamp_ms: int = None
) -> SnowparkDataFrame:
    """
    Filter Snowpark DataFrame for records newer than a given timestamp.

    This function enables incremental ingestion by selecting only records
    with a timestamp greater than the last processed value stored in project
    variables or provided explicitly.

    Args:
        sp_df (SnowparkDataFrame): Input Snowpark DataFrame.
        last_timestamp_ms (int, optional): Timestamp in milliseconds used as
            a lower bound for filtering. If not provided, a project variable
            is used.

    Returns:
        SnowparkDataFrame: Filtered Snowpark DataFrame containing only new records.
    """
    logger.info("Filtering for newer records...")
    timestamp_col_name = cfg.get_project_variable(cfg.TIMESTAMP_COLUMN_VAR_KEY)

    if last_timestamp_ms is None:
        last_timestamp_ms = cfg.get_project_variable(cfg.LAST_FILE_TIMESTAMP_MS_VAR_KEY)

    logger.info(
        "Filtering for records where '%s' > %s",
        timestamp_col_name,
        last_timestamp_ms,
    )

    filtered_sp = sp_df.filter(col(timestamp_col_name) > last_timestamp_ms)
    logger.info("Number of new records (csv files) found: %d", filtered_sp.count())

    return filtered_sp



def _unpack_and_merge_json_column(sp_df: SnowparkDataFrame) -> SnowparkDataFrame:
    """
    Parse, flatten, and extract process signals from a JSON column.

    This function converts a JSON-encoded column containing multiple process
    measurements into a tabular structure by flattening the JSON array and
    extracting selected fields into typed columns.

    Args:
        sp_df (SnowparkDataFrame): Input Snowpark DataFrame containing a JSON column.

    Returns:
        SnowparkDataFrame: Snowpark DataFrame with extracted process features.
    """
    logger.info("Unpacking and extracting JSON data...")

    json_column_name = cfg.get_project_variable(cfg.JSON_DATA_COLUMN_VAR_KEY)
    timestamp_column_name = cfg.get_project_variable(cfg.TIMESTAMP_COLUMN_VAR_KEY)
    device_column_name = cfg.get_project_variable(cfg.DEVICE_COLUMN_VAR_KEY)

    logger.info("Parsing and flattening JSON column: %s", json_column_name)

    sp_df_parsed = sp_df.with_column(
        json_column_name, parse_json(col(json_column_name))
    )

    flattened_df = sp_df_parsed.join_table_function(
        "FLATTEN",
        input=col(json_column_name),
        path=lit(""),
        outer=lit(True),
    )

    json_key_mappings = {
        "PART_NUMBER": {"type": StringType(), "as": "part_number"},
        "PROCESS_CYCLE_RUNNING_SUBPART_NUMBER": {"type": FloatType(), "as": "subpart_no"},
        "C_AXIS_MACHINE_POSITION(0.01deg)": {
            "type": FloatType(),
            "as": "machine_position",
        },
        "THIRD_PARAM_OUTPUT_MONITOR_VALUE(10W)": {"type": FloatType(), "as": "third_param"},
        "FEEDER_SECOND_PARAM(0.01g/sec)": {"type": FloatType(), "as": "second_param"},
        "FEEDER_INTERNAL_FIRST_PARAM(0.01kPa)": {"type": FloatType(), "as": "first_param"},
        "PROCESS_PART_TEMPERATURE_A": {"type": FloatType(), "as": "temp_a"},
        "PROCESS_PART_TEMPERATURE_B": {"type": FloatType(), "as": "temp_b"},
    }

    columns_to_select = [col(timestamp_column_name), col(device_column_name)]

    for json_key, mapping in json_key_mappings.items():
        columns_to_select.append(
            col("VALUE")[json_key].cast(mapping["type"]).as_(mapping["as"])
        )

    logger.info(
        "Extracting %d fields from JSON and preserving %s and %s",
        len(json_key_mappings),
        timestamp_column_name,
        device_column_name
    )

    final_extracted_df = flattened_df.select(*columns_to_select)

    return final_extracted_df



def _lower_column_names(
    df_pd: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert all Pandas DataFrame column names to lowercase.

    Args:
        df_pd (pd.DataFrame): Input Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame with lowercase column names.
    """
    logger.info("Lowercasing Pandas DataFrame column names...")

    df_pd.columns = [col.lower() for col in df_pd.columns]

    return df_pd



def _update_last_file_globals_and_drop_timestamp(
    df: pd.DataFrame,
    update_globals: bool = True
) -> pd.DataFrame:
    """
    Update project variables for incremental ingestion and drop timestamp column.

    The function updates the last processed timestamp and part number stored
    in project-level variables, enabling incremental data loading in future runs.
    The timestamp column is then removed from the DataFrame.

    Args:
        df (pd.DataFrame): Input Pandas DataFrame.
        update_globals (bool): Whether to update project variables.

    Returns:
        pd.DataFrame: DataFrame without the timestamp column.
    """
    logger.info(
        "Updating incremental tracking variables and dropping timestamp column..."
    )

    if df.empty:
        logger.info("No new data processed, skipping variable update and column drop.")
        return df

    part_number_column_name = cfg.get_project_variable(cfg.PART_NUMBER_COLUMN_VAR_KEY)
    timestamp_column_name = cfg.get_project_variable(cfg.TIMESTAMP_COLUMN_VAR_KEY)

    new_last_timestamp = int(df[timestamp_column_name].max())
    new_last_partnumber = None

    if part_number_column_name in df.columns:
        latest_records_at_max_timestamp = df[
            df[timestamp_column_name] == new_last_timestamp
        ]

        if not latest_records_at_max_timestamp.empty:
            new_last_partnumber = latest_records_at_max_timestamp[
                part_number_column_name
            ].iloc[0]
        else:
            logger.warning(
                "No records found at the latest timestamp (%s). Head number "
                "variable will not be updated.",
                new_last_timestamp
            )
    else:
        logger.warning(
            "Head number column '%s' not found. Skipping update for part number variable.",
            part_number_column_name
        )

    if update_globals:
        cfg.modify_project_variable(
            cfg.LAST_FILE_TIMESTAMP_MS_VAR_KEY, new_last_timestamp
        )
        cfg.modify_project_variable(
            cfg.LAST_FILE_PART_NUMBER_VAR_KEY, new_last_partnumber
        )
    else:
        logger.info("Skipping update of Dataiku project variables (test mode).")

    logger.info("Dropping timestamp column '%s' from pandas DataFrame.", timestamp_column_name)

    return df.drop(columns=[timestamp_column_name])



def get_and_prepare_data_from_snowflake(
    snowflake_session: Session,
    test_last_timestamp_ms: int = None,
    test_update_globals: bool = True,
) -> pd.DataFrame:
    """
    Ingest, parse, and prepare production data from Snowflake.

    This is the main orchestration function responsible for:
    - loading raw data from Snowflake,
    - filtering new records based on timestamps,
    - extracting JSON-encoded process signals,
    - updating incremental tracking variables,
    - returning a clean Pandas DataFrame.

    Args:
        snowflake_session (Session): Active Snowflake Snowpark session.
        test_last_timestamp_ms (int, optional): Override timestamp for test runs.
        test_update_globals (bool): Whether to update project variables.

    Returns:
        pd.DataFrame: Prepared Pandas DataFrame ready for feature engineering.

    Raises:
        Exception: Propagates any ingestion or processing error.
    """
    execution_mode = cfg.EXECUTION_MODE
    logger.info("--- Starting Snowflake Data Ingestion: %s ---", execution_mode)

    try:
        initial_sp_df = utils.load_snowflake_table_to_sp(
            snowflake_session,
            cfg.SOURCE_DB_NAME_VAR_KEY,
            cfg.SOURCE_SCHEMA_NAME_VAR_KEY,
            cfg.SOURCE_TABLE_NAME_VAR_KEY,
        )
        logger.info("Initial Snowpark df loaded. Total records: %d", initial_sp_df.count())

        sp_filtered = _return_sp_newer_than_timestamp_files(
            initial_sp_df, last_timestamp_ms=test_last_timestamp_ms
        )

        if sp_filtered.count() == 0:
            logger.info("No new data found since the last run. Returning an empty DataFrame.")
            return pd.DataFrame()

        sp_unpacked = _unpack_and_merge_json_column(sp_filtered)
        pandas_df = sp_unpacked.to_pandas()

        final_pandas_df = pandas_df.pipe(
            _update_last_file_globals_and_drop_timestamp,
            update_globals=test_update_globals,
        ).pipe(_lower_column_names)

        logger.info("Converted to Pandas DataFrame with %d records.", len(final_pandas_df))
        logger.info("--- Snowflake Data Ingestion Completed Successfully: %s ---\n", execution_mode)
        return final_pandas_df

    except Exception as e:
        logger.error("An error occurred during data ingestion: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    # Logging config only for testing purposes
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("--- Running local test of %s module ---", os.path.basename(__file__))
    from dataiku.snowpark import DkuSnowpark

    session = DkuSnowpark().get_session(
        connection_name=cfg.get_project_variable(cfg.SNOWFLAKE_CONNECTION_VAR_KEY)
    )

    try:
        TEST_TIMESTAMP = 1754060000048
        SHOULD_UPDATE_GLOBALS = False

        final_df = get_and_prepare_data_from_snowflake(
            session,
            test_last_timestamp_ms=TEST_TIMESTAMP,
            test_update_globals=SHOULD_UPDATE_GLOBALS,
        )

        logger.info("\n--- Test Run Results ---")
        if not final_df.empty:
            logger.info("Final Pandas DataFrame has %d records.", len(final_df))
            print("DataFrame Info:")
            final_df.info()
            logger.info("DataFrame Head:\n%s", final_df.part())
        else:
            logger.info("No data retrieved. Final DataFrame is empty.")

    except (ValueError, RuntimeError) as e:
        logger.error("TEST FAILED loading data to pandas df. Reason: %s", e, exc_info=True)
