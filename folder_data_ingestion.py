"""
folder_data_ingestion module for Dataiku folders.

This module provides utilities for loading, filtering, and preparing
raw CSV process data stored in Dataiku managed folders. It supports
both OK and NG datasets, applies schema-based column mapping, extracts
metadata from file names, and assigns quality labels to individual
process subparts.
"""

import os
import re
import io
import logging
import __main__
import pandas as pd
import numpy as np

import dataiku

import project_globals as cfg

logger = logging.getLogger(__name__)


def _get_folder_and_list_folder_paths(
    folder_id: str,
    folder_type: str
) -> tuple[dataiku.Folder, list[str]]:
    """
    Retrieve a Dataiku folder and list relevant CSV file paths.

    Optionally filters files based on folder type (e.g. NG).

    Args:
        folder_id (str): Dataiku folder identifier.
        folder_type (str): Type of folder ("NG" or "OK").

    Returns:
        tuple[dataiku.Folder, list[str]]: Dataiku Folder object and
        list of CSV file paths contained in the folder.

    Raises:
        Exception: If folder access or listing fails.
    """
    try:
        folder = dataiku.Folder(folder_id)

        paths = folder.list_paths_in_partition()

        csv_paths = [p for p in paths if p.endswith(".csv")]

        if folder_type == "NG":
            csv_paths = [
                path for path in csv_paths if path.split("/")[-1].split("_")[0] == "A"
            ]

        return folder, csv_paths

    except Exception as e:
        logger.error("ERROR: An error occurred during listing folder files: %s", e)
        raise


def _filter_ng_valves_no(value):
    """
    Validate and normalize NG valve identifiers.

    Accepts either a single digit or a comma-separated list of digits.
    Any other format is treated as invalid.

    Args:
        value (str): Raw NG valve identifier string.

    Returns:
        str or float: Validated NG valve identifier or NaN if invalid.
    """
    if re.match(r"^\d$", value):
        return str(value)
    elif re.match(r"^(\d,)+\d$", value):
        return str(value)
    else:
        return np.nan


def _check_status(row):
    """
    Determine subpart quality status based on NG valve and subpart number.

    Args:
        row (pd.Series): DataFrame row containing subpart and NG valve data.

    Returns:
        str: Quality status label ("NG" or "OK_within_NG").
    """
    if pd.isna(row["ng_valve_no"]):
        return "NG"

    ng_valves_str = str(row["ng_valve_no"]).split(",")
    subpart_no_str = str(row["subpart_no"])

    if any(digit in subpart_no_str for digit in ng_valves_str if digit.isdigit()):
        return "NG"

    return "OK_within_NG"


def _add_column_with_label(df, label):
    """
    Add subpart quality label column to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label (str): Folder type label ("OK" or "NG").

    Returns:
        pd.DataFrame: DataFrame with added `subpart_label` column.
    """
    if label == "OK":
        df["subpart_label"] = "OK"
    else:
        df["subpart_label"] = df.apply(_check_status, axis=1)
    return df


def _combine_files_to_dataframe(
    folder: dataiku.Folder,
    filepaths: list[str],
    files_type: str
) -> pd.DataFrame:
    """
    Load and combine multiple CSV files into a single DataFrame.

    Applies schema-based column selection, extracts metadata from file
    names, assigns NG valve information, and labels subpart quality.

    Args:
        folder (dataiku.Folder): Dataiku folder object.
        filepaths (list[str]): List of CSV file paths.
        files_type (str): Type of files ("NG" or "OK").

    Returns:
        pd.DataFrame: Combined and preprocessed DataFrame.
    """
    header_rows = cfg.get_project_variable(cfg.HEADER_ROWS_VAR_KEY)

    all_dfs = []
    cols_to_use = {}
    temp_index = 10

    schema = list(cfg.get_project_variable(cfg.RAW_DATA_SCHEMA_KEY))
    feature_cols = [col for col in schema if col not in ["device", "part_number"]]

    for index, col in enumerate(feature_cols):
        if "temp" in col:
            cols_to_use[temp_index] = col
            temp_index += 1
        else:
            cols_to_use[index] = col

    for file_path in filepaths:
        with folder.get_download_stream(file_path) as byte_stream:
            text_stream = io.TextIOWrapper(byte_stream, encoding="latin1")
            text_data = text_stream.read()

            df = pd.read_csv(
                io.StringIO(text_data),
                skiprows=header_rows + 1,
                usecols=cols_to_use.keys(),
                names=cols_to_use.values(),
            )

            full_file_name = os.path.basename(file_path).split("_")
            if len(full_file_name) > 3:
                part_number = full_file_name[3]
                df.insert(0, "device", cfg.MACHINE_NAME_KEY)
                df.insert(1, "part_number", str(part_number))

                if files_type == "NG":
                    ng_valve_no = full_file_name[-1].split(".")[0]
                    df["ng_valve_no"] = _filter_ng_valves_no(ng_valve_no)
                else:
                    df["ng_valve_no"] = np.nan
            else:
                continue

            df = _add_column_with_label(df, files_type)
            all_dfs.append(df)

    final_df = pd.concat(all_dfs)

    logger.info("Casting feature columns to float...")
    for col in feature_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(
                final_df[col],
                errors="coerce",
                downcast="float"
            )
        else:
            logger.warning("Column '%s' not found in final DataFrame for casting.", col)

    return final_df


def get_and_prepare_data_from_folder(
    folder_id: str,
    folder_type: str,
) -> pd.DataFrame:
    """
    Load and prepare process data from a Dataiku folder.

    Orchestrates file discovery, CSV loading, data concatenation,
    type casting, and quality labeling.

    Args:
        folder_id (str): Dataiku folder identifier.
        folder_type (str): Type of folder ("NG" or "OK").

    Returns:
        pd.DataFrame: Prepared pandas DataFrame with process data.
    """
    main_filename = cfg.EXECUTION_MODE

    logger.info(
        "--- Starting Folder Data Ingestion from %s folder: %s ---",
        folder_type,
        main_filename,
    )

    try:
        folder, filepaths = _get_folder_and_list_folder_paths(folder_id, folder_type)

        logger.info(
            "Folder (%s) contains: %d files", folder_type, len(filepaths)
        )

        if len(filepaths) == 0:
            logger.info(
                "No files found in (%s) folder. Returning an empty DataFrame.",
                folder_type,
            )
            return pd.DataFrame()

        logger.info("Combining csv files to pandas DataFrame")

        df = _combine_files_to_dataframe(folder, filepaths, folder_type)

        logger.info("Files sucessfully converted into pandas DataFrame")
        logger.info(
            "--- Folder Data Ingestion from (%s) folder Completed Successfully: %s ---\n",
            folder_type,
            main_filename,
        )

        return df

    except Exception as e:
        logger.error(
            "ERROR: An error occurred during data ingestion and preparation: %s", e
        )
        raise


if __name__ == "__main__":
    print(f"--- Running local test of {os.path.basename(__file__)} module ---")

    ng_folder_id = cfg.get_project_variable(cfg.TRAIN_NG_FOLDER_ID_KEY)
    ok_folder_id = cfg.get_project_variable(cfg.TRAIN_OK_FOLDER_ID_KEY)

    ng_df = get_and_prepare_data_from_folder(ng_folder_id, "NG")
    ok_df = get_and_prepare_data_from_folder(ok_folder_id, "OK")

    print(ng_df.info())
    print(ok_df.info())

    print(ng_df.head())
    print(ok_df.head())
