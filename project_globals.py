"""
projec_globals module.

This module defines constants and helper functions used to manage
Dataiku project variables related to a production process pipeline.
It focuses on configuration, state tracking, and metadata exchange
between data ingestion, feature engineering, training, and scoring
stages of a machine learning workflow operating on manufacturing data.
"""

import logging
import dataiku

MACHINE_NAME_KEY = "MACHINE_NAME"

LAST_FILE_TIMESTAMP_MS_VAR_KEY = "LAST_FILE_TIMESTAMP_MS"
LAST_FILE_PART_NUMBER_VAR_KEY = "LAST_FILE_PART_NUMBER"
LAST_FILE_DEVICE_VAR_KEY = "LAST_FILE_DEVICE"

SOURCE_DB_NAME_VAR_KEY = "SOURCE_DB_NAME"
SOURCE_SCHEMA_NAME_VAR_KEY = "SOURCE_SCHEMA_NAME"
SOURCE_TABLE_NAME_VAR_KEY = "SOURCE_TABLE_NAME"

INTERMEDIATE_DB_NAME_VAR_KEY = "INTERMEDIATE_DB_NAME"
INTERMEDIATE_SCHEMA_NAME_VAR_KEY = "INTERMEDIATE_SCHEMA_NAME"
INTERMEDIATE_MODEL_OUTPUT_TABLE_NAME_VAR_KEY = "INTERMEDIATE_MODEL_OUTPUT_TABLE_NAME"
INTERMEDIATE_MODEL_PERFORMANCE_TEST_VAR_KEY = "INTERMEDIATE_MODEL_PERFORMANCE_TEST_TABLE_NAME"
INTERMEDIATE_RAW_SCORING_TABLE_NAME_VAR_KEY = "INTERMEDIATE_RAW_SCORING_TABLE_NAME"
INTERMEDIATE_RAW_TRAIN_TABLE_NAME_VAR_KEY = "INTERMEDIATE_RAW_TRAIN_TABLE_NAME"
INTERMEDIATE_RAW_TEST_TABLE_NAME_VAR_KEY = "INTERMEDIATE_RAW_TEST_TABLE_NAME"
INTERMEDIATE_RAW_VAL_TABLE_NAME_VAR_KEY = "INTERMEDIATE_RAW_VAL_TABLE_NAME"
INTERMEDIATE_PREPARED_SCORING_TABLE_NAME_VAR_KEY = "INTERMEDIATE_PREPARED_SCORING_TABLE_NAME"
INTERMEDIATE_PREPARED_TRAIN_TABLE_NAME_VAR_KEY = "INTERMEDIATE_PREPARED_TRAIN_TABLE_NAME"
INTERMEDIATE_PREPARED_TEST_TABLE_NAME_VAR_KEY = "INTERMEDIATE_PREPARED_TEST_TABLE_NAME"
INTERMEDIATE_PREPARED_VAL_TABLE_NAME_VAR_KEY = "INTERMEDIATE_PREPARED_VAL_TABLE_NAME"

TIMESTAMP_COLUMN_VAR_KEY = "TIMESTAMP_COLUMN_NAME"
JSON_DATA_COLUMN_VAR_KEY = "JSON_COLUMN_NAME"
PART_NUMBER_COLUMN_VAR_KEY = "PART_NUMBER_COLUMN_NAME"
HEADER_ROWS_VAR_KEY = "HEADER_CSV_ROWS"
DEVICE_COLUMN_VAR_KEY = "DEVICE_COLUMN_NAME"

SNOWFLAKE_CONNECTION_VAR_KEY = "SNOWFLAKE_CONNECTION_NAME"
RAW_DATA_SCHEMA_KEY = "RAW_DATA_SCHEMA"
FEATUE_ENG_SCHEMA_KEY = "FEATUE_ENG_SCHEMA"

MAX_TRAIN_SEQ_LENGTH_VAR_KEY = "MAX_TRAIN_SEQ_LENGTH"
LAST_SCALER_FILENAME_KEY = "LAST_SCALER_FILENAME"
LAST_WEIGHTS_FOR_SAMPLER_FILENAME_KEY = "LAST_WEIGHTS_FOR_SAMPLER_FILENAME"
LAST_WEIGHTS_FOR_SAMPLER_KEY = "LAST_WEIGHTS_FOR_SAMPLER"

TRAIN_OK_FOLDER_ID_KEY = "TRAIN_OK_FOLDER_ID"
TRAIN_NG_FOLDER_ID_KEY = "TRAIN_NG_FOLDER_ID"
SAVED_MODELS_FOLDER_ID_KEY = "SAVED_MODELS_FOLDER_ID"
BEST_CNN_MODEL_PARAMS_KEY = "BEST_CNN_MODEL_PARAMS"
TRAINING_PARAMS_KEY = "TRAINING_PARAMS"
BEST_MODEL_FILENAME_KEY = "BEST_MODEL_FILENAME"

EXECUTION_MODE = None

logger = logging.getLogger(__name__)


def modify_project_variable(variable_name_key: str, new_value):
    """
    Safely updates a Dataiku project standard variable.

    The function retrieves current project variables, compares the existing
    value with the new one, and updates the variable only if a change is
    required. All operations are logged for traceability.

    Args:
        variable_name_key (str): Name of the Dataiku project variable to update.
        new_value (Any): New value to be assigned to the project variable.

    Raises:
        Exception: If the variable cannot be updated due to API or permission issues.
    """
    logger.debug("Attempting to modify project variable '%s'...", variable_name_key)
    try:
        client = dataiku.api_client()
        project = client.get_project(dataiku.default_project_key())
        project_vars = project.get_variables()
        old_value = project_vars.get("standard", {}).get(variable_name_key)

        if str(old_value) == str(new_value):
            logger.info(
                "Value for '%s' is already '%s'. No update needed.",
                variable_name_key,
                new_value,
            )
            return

        logger.info(
            "Changing '%s' from '%s' to '%s'", variable_name_key, old_value, new_value
        )
        project_vars["standard"][variable_name_key] = new_value
        project.set_variables(project_vars)
        logger.info("Successfully updated variable '%s'.", variable_name_key)

    except Exception as e:
        logger.error(
            "Failed to modify project variable '%s'. Reason: %s",
            variable_name_key,
            e,
            exc_info=True,
        )
        raise


def get_project_variable(variable_name: str):
    """
    Retrieves a Dataiku project standard variable in a safe manner.

    The function accesses the current Dataiku project configuration and
    returns the value of the requested variable. If the variable is missing,
    an explicit error is raised to prevent silent misconfiguration.

    Args:
        variable_name (str): Name of the Dataiku project variable to retrieve.

    Returns:
        Any: Value of the requested project variable.

    Raises:
        ValueError: If the variable is not defined in the project.
        Exception: If an unexpected error occurs while accessing project variables.
    """
    logger.debug("Attempting to get project variable '%s'...", variable_name)
    try:
        client = dataiku.api_client()
        project = client.get_project(dataiku.default_project_key())
        project_vars = project.get_variables()

        value = project_vars.get("standard", {}).get(variable_name)

        if value is not None:
            logger.debug("Successfully retrieved variable '%s'.", variable_name)
            return value
        else:
            error_msg = (
                f"Required Dataiku Project Variable '{variable_name}' is missing. "
                "Please set it in Variables tab."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    except Exception as e:
        logger.error(
            "Failed to get project variable '%s'. An unexpected error occurred: %s",
            variable_name,
            e,
            exc_info=True,
        )
        raise
