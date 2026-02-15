"""
utils module

Feature engineering utilities for processing production process data.

This module provides helper functions to interact with Snowflake/Snowpark,
convert between Pandas and Snowpark DataFrames, and prepare data for 
further processing or storage. It is primarily designed for handling 
process data from manufacturing or production lines.

Module: feature_engineering
"""

import logging
import pandas as pd

from dataiku.snowpark import DkuSnowpark
from snowflake.snowpark import Session, DataFrame as SnowparkDataFrame
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import col

import project_globals as cfg

logger = logging.getLogger(__name__)


def create_snowflake_session(
    connection_name_key: str
) -> Session:
    """Create and return a Snowpark session object for the specified connection.

    Args:
        connection_name_key (str): Dataiku project variable key storing the Snowflake connection name.

    Returns:
        Session: Snowpark Session object.

    Raises:
        ValueError: If the connection name is invalid.
        Exception: For unexpected errors during session creation.
    """
    logger.info("Creating Snowpark session...")
    try:
        connection_name = cfg.get_project_variable(connection_name_key)

        sp_handler = DkuSnowpark()
        session = sp_handler.get_session(connection_name=connection_name)

        logger.info("Snowpark session established successfully.")
        return session
    except ValueError as e:
        logger.error("Configuration error: Could not create Snowpark session. Reason: %s", e)
        raise
    except Exception:
        logger.error("Failed to create Snowpark session due to an unexpected error.", exc_info=True)
        raise


def load_snowflake_table_to_sp(
    snowflake_session: Session,
    db_var_key: str,
    schema_var_key: str,
    table_var_key: str,
) -> SnowparkDataFrame:
    """Load a Snowflake table as a Snowpark DataFrame reference.

    Args:
        snowflake_session (Session): Active Snowpark session.
        db_var_key (str): Project variable key for the database name.
        schema_var_key (str): Project variable key for the schema name.
        table_var_key (str): Project variable key for the table name.

    Returns:
        SnowparkDataFrame: Reference to the Snowflake table.

    Raises:
        ValueError: If one or more project variables are missing.
        RuntimeError: If SQL or other unexpected errors occur.
    """
    try:
        db = cfg.get_project_variable(db_var_key)
        schema = cfg.get_project_variable(schema_var_key)
        table = cfg.get_project_variable(table_var_key)

        if None in [db, schema, table]:
            error_msg = """Failed to load Snowpark table:
                        Missing one or more required Dataiku proj vars."""
            logger.error(error_msg)
            raise ValueError(error_msg)

        full_table_name = f'"{db}"."{schema}"."{table}"'

        logger.info("Creating Snowpark DataFrame reference for: %s", full_table_name)

        df_table = snowflake_session.table(full_table_name)

        logger.info("Successfully created Snowpark DataFrame reference for table '%s'.", table)
        return df_table

    except SnowparkSQLException as e:
        logger.error(
            "SQL error while loading Snowpark table '%s'. Check if table exists and permissions.",
            full_table_name,
            exc_info=True,
        )
        raise RuntimeError(f"Failed to load Snowpark table '{full_table_name}'") from e

    except Exception as e:
        logger.error(
            "An unexpected error occurred while loading Snowpark table '%s'.",
            full_table_name,
            exc_info=True,
        )
        raise RuntimeError(
            f"Error loading Snowpark table '{db}.{schema}.{table}': {e}"
        ) from e

    return df_table


def prepare_pandas_df_for_snowflake_table(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Prepare a Pandas DataFrame for Snowflake upload.

    Adds a 'row_index' column to preserve row order during transfer.

    Args:
        df (pd.DataFrame): Input Pandas DataFrame.

    Returns:
        pd.DataFrame: Prepared DataFrame with 'row_index' column.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping preparation.")
        return df

    df_prepared = df.copy()

    df_prepared = df_prepared.reset_index(drop=True).reset_index()
    df_prepared = df_prepared.rename(columns={'index': 'row_index'})
    logger.info("Added 'row_index' column to preserve row order.")

    # # 3. Ujednolić nazwy kolumn do wielkich liter
    # df_prepared.columns = [col.upper() for col in df_prepared.columns]
    # logger.debug("Converted all column names to uppercase.")

    return df_prepared


def prepare_sp_for_snowflake_table(
    sp: SnowparkDataFrame
) -> SnowparkDataFrame:
    """Prepare a Snowpark DataFrame for Snowflake upload.

    Sorts the DataFrame by 'row_index' descending to preserve order.

    Args:
        sp (SnowparkDataFrame): Input Snowpark DataFrame.

    Returns:
        SnowparkDataFrame: Prepared Snowpark DataFrame.
    """
    sp_prepared = sp.sort(col('"row_index"').desc())

    return sp_prepared


def save_pandas_df_to_snowpark_table(
    df: pd.DataFrame,
    snowflake_session: Session,
    db_var_key: str,
    schema_var_key: str,
    table_var_key: str,
    write_mode: str = "overwrite",
) -> SnowparkDataFrame:
    """Save a Pandas DataFrame to a Snowflake table via Snowpark.

    Args:
        df (pd.DataFrame): Input DataFrame.
        snowflake_session (Session): Active Snowpark session.
        db_var_key (str): Project variable key for the database.
        schema_var_key (str): Project variable key for the schema.
        table_var_key (str): Project variable key for the table.
        write_mode (str, optional): "overwrite" or "append". Defaults to "overwrite".

    Returns:
        SnowparkDataFrame: Reference to the saved table.

    Raises:
        ValueError, SnowparkSQLException, Exception: For errors during saving.
    """
    df_to_save = prepare_pandas_df_for_snowflake_table(df)

    try:
        db = cfg.get_project_variable(db_var_key)
        schema = cfg.get_project_variable(schema_var_key)
        table = cfg.get_project_variable(table_var_key)

        if None in [db, schema, table]:
            error_msg = """Failed to resolve table path:
                        Missing one or more required Dataiku Project Variables."""
            logger.error(error_msg)
            raise ValueError(error_msg)

        full_table_name = f'"{db}"."{schema}"."{table}"'

        logger.info("Preparing to save Pandas DataFrame to target table: %s, "
                    "Write mode: '%s'", full_table_name, write_mode)

        if df_to_save.empty and write_mode != "overwrite":
            logger.info(
                "Input DataFrame is empty and write mode is not 'overwrite'. No action taken."
            )
        else:
            if df_to_save.empty:
                logger.info(
                    "Input DataFrame is empty. `write_pandas` will create/overwrite an empty table."
                )

            snowflake_session.write_pandas(
                df_to_save,
                table_name=table,
                database=db,
                schema=schema,
                auto_create_table=True,
                overwrite=(write_mode == "overwrite"),
            )

            logger.info("Successfully wrote %d Pandas DataFrame rows "
                        "to %s.", len(df_to_save), full_table_name)

        logger.info("Returning Snowpark DataFrame reference to '%s'.", full_table_name)
        return snowflake_session.table(full_table_name)

    except (ValueError, SnowparkSQLException) as e:
        logger.error(
            """A predictable error occurred while saving Snowpark DataFrame data to Snowflake.
            Check configuration or table state. Reason: %s""",
            e,
            exc_info=True,
        )
        raise

    except Exception as e:
        logger.error(
            "An unexpected error occurred while saving Snowpark DataFrame data to Snowflake: %s",
            e,
            exc_info=True,
        )
        raise


def save_sp_to_snowpark_table(
    sp: SnowparkDataFrame,
    snowflake_session: Session,
    db_var_key: str,
    schema_var_key: str,
    table_var_key: str,
    write_mode: str = "overwrite",
):
    """Save a Snowpark DataFrame to a Snowflake table.

    Args:
        sp (SnowparkDataFrame): Input Snowpark DataFrame.
        snowflake_session (Session): Active Snowpark session.
        db_var_key (str): Project variable key for database.
        schema_var_key (str): Project variable key for schema.
        table_var_key (str): Project variable key for table.
        write_mode (str, optional): 'overwrite' or 'append'. Defaults to 'overwrite'.

    Returns:
        SnowparkDataFrame: Reference to saved table.

    Raises:
        ValueError, SnowparkSQLException, Exception: For errors during saving.
    """
    sp_to_save = prepare_sp_for_snowflake_table(sp)

    try:
        db = cfg.get_project_variable(db_var_key)
        schema = cfg.get_project_variable(schema_var_key)
        table = cfg.get_project_variable(table_var_key)

        if None in [db, schema, table]:
            error_msg = """Failed to resolve table path:
                        Missing one or more required Dataiku Project Variables."""
            logger.error(error_msg)
            raise ValueError(error_msg)

        full_table_name = f'"{db}"."{schema}"."{table}"'

        logger.info("Preparing to save Snowpark DataFrame to target table: %s, "
                    "Write mode: '%s'", full_table_name, write_mode)

        row_count = sp_to_save.count()

        if row_count == 0 and write_mode != "overwrite":
            logger.info(
                "Input Snowpark DataFrame is empty and write mode is not 'overwrite'. "
                "No action taken."
            )
            return snowflake_session.table(full_table_name)

        else:
            if row_count == 0:
                logger.info(
                    "Input DataFrame is empty. `write_pandas` will create/overwrite an empty table."
                )

            sp_to_save.write.mode(write_mode).save_as_table(full_table_name)

            logger.info("Successfully wrote %d Snowpark Dataframe rows "
                        "to %s.", row_count, full_table_name)

            return snowflake_session.table(full_table_name)

    except (ValueError, SnowparkSQLException) as e:
        logger.error(
            """A predictable error occurred while saving data to Snowflake.
            Check configuration or table state. Reason: %s""",
            e,
            exc_info=True,
        )
        raise

    except Exception as e:
        logger.error(
            "An unexpected error occurred while saving data to Snowflake: %s",
            e,
            exc_info=True,
        )
        raise


def convert_sp_to_pandas(
    sp: SnowparkDataFrame,
    sort_col_name: str = "row_index"
) -> pd.DataFrame:
    """Convert a Snowpark DataFrame to a Pandas DataFrame, optionally sorting by a column.

    Args:
        sp (SnowparkDataFrame): Input Snowpark DataFrame.
        sort_col_name (str, optional): Column name to sort by. Defaults to 'row_index'.

    Returns:
        pd.DataFrame: Converted and sorted Pandas DataFrame.

    Raises:
        SnowparkSQLException: If sorting column does not exist in Snowpark.
        Exception: For other unexpected errors during conversion.
    """
    logger.info(
        "Converting Snowpark DataFrame to Pandas DataFrame, sorting by '%s' ...",
        sort_col_name
    )

    try:
        if sp.count() == 0:
            logger.warning("Input Snowpark DataFrame is empty. Returning empty Pandas DataFrame.")
            return pd.DataFrame()

        sorted_sp = sp.sort(col(f'"{sort_col_name}"'))

        pandas_df = sorted_sp.to_pandas()

        sorted_pandas_df = pandas_df.sort_values(by=sort_col_name)

        logger.info(
            "Successfully converted to pandas DataFrame with %d rows.",
            len(sorted_pandas_df)
        )

        return sorted_pandas_df

    except SnowparkSQLException as e:
        logger.error(
            "A Snowpark SQL error occurred during conversion to pandas. "
            "Check if sort column '%s' exists. Reason: %s",
            sort_col_name, e, exc_info=True
        )
        raise

    except Exception as e:
        logger.error(
            "An unexpected error occurred during conversion to pandas: %s",
            e, exc_info=True
        )
        raise
