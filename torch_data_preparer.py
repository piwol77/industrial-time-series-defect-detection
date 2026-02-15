"""
torch_data_preparer module

This module provides utilities for preparing time-series production process
data for PyTorch modeling. It includes:

- Custom PyTorch Dataset classes for sequence and test data.
- Functions to create sequences with metadata and labels.
- Reshaping utilities between 2D and 3D arrays.
- Feature scaling with sklearn's StandardScaler.
- WeightedRandomSampler creation for imbalanced datasets.
- Functions to prepare DataLoaders for training, validation, and testing.

The module is designed for sequential process data with per-part and per-subpart
measurements from industrial additive manufacturing or similar processes.
"""

import logging
import os
import io
from datetime import datetime
import __main__
import joblib

import pandas as pd
import numpy as np

import dataiku

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

import project_globals as cfg

logger = logging.getLogger(__name__)

class SeatSequence(Dataset):
    """
    PyTorch Dataset for sequences of features and labels.

    Each item is a tuple of (features, label) corresponding to one sequence.

    Args:
        x (np.ndarray): Feature sequences of shape (samples, time_steps, features).
        y (np.ndarray): Corresponding labels of shape (samples,).
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TestDataset(Dataset):
    """
    PyTorch Dataset for sequences in scoring or testing mode with metadata.

    Each item is a tuple of (features, dummy_label, metadata).

    Args:
        x (np.ndarray): Feature sequences of shape (samples, time_steps, features).
        y (np.ndarray): Dummy labels (all zeros, for compatibility with PyTorch).
        metadata (np.ndarray): Metadata per sequence (e.g., part_number identifiers).
    """
    def __init__(self, x, y, metadata):
        self.x = x
        self.y = y
        self.metadata = metadata

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.metadata[idx]


def _reshape_3d_to_2d(data: np.ndarray) -> np.ndarray:
    """
    Flatten a 3D array (samples, time_steps, features) into 2D (samples*time_steps, features).

    Args:
        data (np.ndarray): Input 3D array.

    Returns:
        np.ndarray: Flattened 2D array.

    Raises:
        ValueError: If input is not 3-dimensional.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3-dimensional")
    samples, time_steps, features = data.shape
    return data.reshape(samples * time_steps, features)


def _reshape_2d_to_3d(data: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Restore a 2D array to its original 3D shape.

    Args:
        data (np.ndarray): Flattened 2D array (samples*time_steps, features).
        original_shape (tuple): Original shape (samples, time_steps, features).

    Returns:
        np.ndarray: Reshaped 3D array.

    Raises:
        ValueError: If original_shape does not have 3 dimensions.
    """
    if len(original_shape) != 3:
        raise ValueError("Original shape must be a tuple of 3 dimensions")
    samples, time_steps, features = original_shape
    return data.reshape(samples, time_steps, features)


def create_sequences_with_metadata(
    df: pd.DataFrame,
    faeture_eng_schema_key: str,
    part_number_column_key: str,
    seq_length_key: str,
    execution_mode: str,
    subpart_column: str = 'subpart_no',
    label_column: str = 'subpart_label'
):
    """
    Generate sequences of features with corresponding metadata and labels.

    Args:
        df (pd.DataFrame): Input Pandas DataFrame containing features, labels, and metadata.
        faeture_eng_schema_key (str): Project variable key for feature column schema.
        part_number_column_key (str): Project variable key for part identifier column.
        seq_length_key (str): Project variable key for sequence length.
        execution_mode (str): Execution mode, e.g., 'train', 'val', 'score'.
        subpart_column (str): Column representing subpart number. Default is 'subpart_no'.
        label_column (str): Column representing label. Default is 'subpart_label'.

    Returns:
        Tuple[np.ndarray]: 
            - In scoring mode: (x_sequences, metadata)
            - In training/validation mode: (x_sequences, metadata, y_labels, groups)
    """
    schema = [s.lower() for s in list(cfg.get_project_variable(faeture_eng_schema_key))]
    metadata_column = (cfg.get_project_variable(part_number_column_key)).lower()
    seq_length = int(cfg.get_project_variable(seq_length_key))

    metadata_columns = ['row_index', 'device', 'part_index', 'subpart_index'] + [metadata_column]
    feature_columns = [col for col in schema if col not in metadata_columns]

    sequences, metadata = [], []
    is_scoring_mode = 'score' in execution_mode

    labels, groups = ([], []) if not is_scoring_mode else (None, None)

    for part_name, group_df in df.groupby(metadata_column):
        data = group_df[feature_columns].values

        if not is_scoring_mode:
            label_values = group_df[label_column].values
            subpart_values = group_df[subpart_column].values

        for i in range(0, len(data) - seq_length + 1, seq_length):
            sequences.append(data[i:i + seq_length])
            metadata.append(part_name)

            if not is_scoring_mode:
                labels.append(label_values[i + seq_length - 1])
                group_id = f"{part_name}#{subpart_values[i + seq_length - 1]}"
                groups.append(group_id)

    if is_scoring_mode:
        return np.array(sequences), np.array(metadata)
    else:
        return np.array(sequences), np.array(metadata), np.array(labels), np.array(groups)


def fit_and_save_scaler(x_train: np.ndarray, folder_id: str, timestamp: str) -> tuple[StandardScaler, str]:
    """
    Fit a StandardScaler to training data and save it to a Dataiku folder.

    Args:
        x_train (np.ndarray): Training data (3D array: samples, time_steps, features).
        folder_id (str): Dataiku folder ID for saving the scaler.
        timestamp (str): Timestamp string used for filename.

    Returns:
        Tuple[StandardScaler, str]: Fitted scaler and filename used for saving.

    Raises:
        ValueError: If input x_train is not 3D.
        Exception: If saving to Dataiku folder fails.
    """
    if x_train.ndim != 3:
        raise ValueError("Input training data must be a 3D numpy array.")

    logger.info("Fitting StandardScaler on training data...")
    data_2d = _reshape_3d_to_2d(x_train)

    scaler = StandardScaler()
    scaler.fit(data_2d)

    logger.info("Scaler fitted successfully.")

    filename = f"standard_scaler_{timestamp}.joblib"

    try:
        folder = dataiku.Folder(folder_id)
        with folder.get_writer(filename) as writer:
            buffer = io.BytesIO()
            joblib.dump(scaler, buffer)
            writer.write(buffer.getvalue())
        logger.info("Fitted StandardScaler saved to Dataiku folder as '%s'.", filename)
    except Exception as e:
        logger.error("Failed to save the scaler: %s", e, exc_info=True)
        raise

    return scaler, filename


def load_scaler_from_folder(folder_id: str, filename: str) -> StandardScaler:
    """
    Load a previously saved StandardScaler from a Dataiku folder.

    Args:
        folder_id (str): Dataiku folder ID.
        filename (str): Name of the scaler file to load.

    Returns:
        StandardScaler: Loaded StandardScaler object.

    Raises:
        Exception: If loading from Dataiku folder fails.
    """
    logger.info("Loading StandardScaler from folder...")
    try:
        folder = dataiku.Folder(folder_id)
        with folder.get_download_stream(filename) as reader:
            scaler = joblib.load(io.BytesIO(reader.read()))

        logger.info("StandardScaler loaded successfully.")
        return scaler

    except Exception as e:
        logger.error("Failed to load the scaler: %s", e, exc_info=True)
        raise


def transform_data_using_scaler(data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply a fitted StandardScaler to 3D data.

    Args:
        data (np.ndarray): Input data (3D array: samples, time_steps, features).
        scaler (StandardScaler): Fitted scaler to apply.

    Returns:
        np.ndarray: Scaled 3D data.
    """
    logger.info("Transforming data using StandardScaler...")
    data_2d = _reshape_3d_to_2d(data)
    data_scaled_2d = scaler.transform(data_2d)
    return _reshape_2d_to_3d(data_scaled_2d, data.shape)


def save_weights_for_sampler(y_train_tensor: torch.tensor) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to balance imbalanced classes in training.

    Args:
        y_train_tensor (torch.tensor): Tensor of labels for training sequences.

    Returns:
        WeightedRandomSampler: Sampler to be used in PyTorch DataLoader.
    """
    y_train_np = y_train_tensor.numpy().astype(int)
    class_counts = np.bincount(y_train_np)
    weights_per_class = 1.0 / class_counts

    sample_weights = np.array([weights_per_class[label] for label in y_train_np])
    sample_weights = torch.tensor(sample_weights, dtype = torch.float32)

    weighted_sampler = WeightedRandomSampler(
        weights = sample_weights, num_samples = len(sample_weights), replacement = True
    )

    cfg.modify_project_variable(cfg.LAST_WEIGHTS_FOR_SAMPLER_KEY, weights_per_class.tolist())

    return weighted_sampler


def prepare_torch_data(df: pd.DataFrame, data_type: str, batch_size: int = 64) -> DataLoader:
    """
    Prepare a PyTorch DataLoader from a Pandas DataFrame.

    Handles training, validation, or scoring modes:
    - Creates sequences with metadata.
    - Applies scaling using StandardScaler.
    - Adds weighted sampling for imbalanced training.
    - Returns appropriate Dataset and DataLoader.

    Args:
        df (pd.DataFrame): Input Pandas DataFrame.
        data_type (str): Type of data: 'train', 'val', or 'test'.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader ready for model training or scoring.

    Raises:
        ValueError: If data_type is unknown or DataFrame is empty in non-score mode.
        Exception: For any preprocessing or Dataiku folder errors.
    """
    execution_mode = cfg.EXECUTION_MODE
    logger.info("--- Preparing torch DataLoader [mode=%s, data_type=%s] ---",
                execution_mode, data_type)

    if df.empty:
        logger.warning("Input DataFrame is empty, skipping processing.")
        return DataLoader(Dataset())

    try:
        folder_id = cfg.get_project_variable(cfg.SAVED_MODELS_FOLDER_ID_KEY)

        result_tuple = create_sequences_with_metadata(
            df,
            cfg.FEATUE_ENG_SCHEMA_KEY,
            cfg.PART_NUMBER_COLUMN_VAR_KEY,
            cfg.MAX_TRAIN_SEQ_LENGTH_VAR_KEY,
            execution_mode
        )

        if 'score' in execution_mode:
            x, metadata = result_tuple
            scaler_filename = cfg.get_project_variable(cfg.LAST_SCALER_FILENAME_KEY)
            scaler = load_scaler_from_folder(folder_id, scaler_filename)

            x_scaled = transform_data_using_scaler(x, scaler)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            dataset = TestDataset(x_tensor, np.zeros(len(x_tensor), dtype=np.float32), metadata)
            return DataLoader(dataset, shuffle=False, batch_size=batch_size)

        else:
            x, metadata, y, groups = result_tuple
            train_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        if 'train' in data_type:
            scaler, scaler_filename = fit_and_save_scaler(x, folder_id, train_timestamp)
            cfg.modify_project_variable(cfg.LAST_SCALER_FILENAME_KEY, scaler_filename)
        else:
            scaler_filename = cfg.get_project_variable(cfg.LAST_SCALER_FILENAME_KEY)
            scaler = load_scaler_from_folder(folder_id, scaler_filename)

        x_scaled = transform_data_using_scaler(x, scaler)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if 'train' in data_type:
            weighted_sampler = save_weights_for_sampler(y_tensor)
            dataset = SeatSequence(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, sampler=weighted_sampler, batch_size=batch_size)
        elif 'val' in data_type:
            dataset = SeatSequence(x_tensor, y_tensor)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        elif 'test' in data_type:
            dataset = TestDataset(x_tensor, y_tensor, metadata)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        else:
            error_msg = f"Unknown data_type: '{data_type}'. Must contain 'train', 'val', or 'test'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("--- Preparing DataLoader from Pandas df Completed Successfully: %s ---\n",
            execution_mode)

        return dataloader

    except Exception as e:
        logger.error("An error occurred during data pre-processing: %s", e, exc_info=True)
        raise
