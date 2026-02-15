"""
model_handler module.

This module defines neural network architectures and utilities for
loading trained PyTorch models used in binary classification of
industrial process data. It supports convolutional and recurrent
(sequence-based) models and integrates with Dataiku folders for
model state persistence.
"""

import logging
import io

import dataiku

import torch
import torch.nn as nn

import project_globals as cfg

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """
    One-dimensional convolutional neural network for time-series data.

    This model applies stacked 1D convolutional blocks with optional
    batch normalization and pooling, followed by global pooling and
    a fully connected classification part.
    """

    def __init__(self,
                 input_channels,
                 num_filters=64,
                 kernel_sizes=None,
                 num_layers=3,
                 dropout=0.2,
                 use_pool=True,
                 pool_type="max",
                 use_batchnorm=True,
                 activation="ReLU",
                 leaky_slope=0.01
                ):
        """
        Initialize the CNNModel.

        Args:
            input_channels (int): Number of input feature channels.
            num_filters (int): Number of filters in the first convolutional layer.
            kernel_sizes (list[int], optional): Kernel sizes for each convolutional layer.
            num_layers (int): Number of convolutional blocks.
            dropout (float): Dropout probability before the final classifier.
            use_pool (bool): Whether to apply pooling layers.
            pool_type (str): Type of pooling ("max" or "avg").
            use_batchnorm (bool): Whether to use batch normalization.
            activation (str): Activation function name.
            leaky_slope (float): Negative slope for LeakyReLU activation.
        """
        super().__init__()

        if activation == "ReLU":
            act = nn.ReLU()
        elif activation == "LeakyReLU":
            act = nn.LeakyReLU(leaky_slope)
        elif activation == "ELU":
            act = nn.ELU()
        else:
            act = nn.SELU()

        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        self.blocks = nn.ModuleList()
        in_ch = input_channels
        out_ch = num_filters
        for i in range(num_layers):
            k = kernel_sizes[i]
            conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2)
            layers = [conv]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(act)
            if use_pool:
                if pool_type == "max":
                    layers.append(nn.MaxPool1d(2))
                else:
                    layers.append(nn.AvgPool1d(2))
            self.blocks.append(nn.Sequential(*layers))
            in_ch, out_ch = out_ch, out_ch * 2

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, apply_sigmoid=True):
        """
        Perform a forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            apply_sigmoid (bool): Whether to apply sigmoid activation
                to the output.

        Returns:
            torch.Tensor: Model output logits or probabilities.
        """
        x = x.transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x) if apply_sigmoid else x


class LSTMModel(nn.Module):
    """
    LSTM-based neural network for sequential data classification.

    This model processes time-series data using stacked LSTM layers,
    followed by layer normalization and a fully connected output layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        """
        Initialize the LSTMModel.

        Args:
            input_dim (int): Number of input features per time step.
            hidden_dim (int): Number of hidden units in LSTM layers.
            output_dim (int): Output dimension.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, apply_sigmoid=True):
        """
        Perform a forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            apply_sigmoid (bool): Whether to apply sigmoid activation
                to the output.

        Returns:
            torch.Tensor: Model output logits or probabilities.
        """
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        out = self.fc(out)

        if apply_sigmoid:
            return self.sigmoid(out)
        else:
            return out


class GRUModel(nn.Module):
    """
    GRU-based neural network for sequential data classification.

    This model uses GRU layers followed by layer normalization and
    a fully connected output layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        """
        Initialize the GRUModel.

        Args:
            input_dim (int): Number of input features per time step.
            hidden_dim (int): Number of hidden units in GRU layers.
            output_dim (int): Output dimension.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout probability between GRU layers.
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, apply_sigmoid=True):
        """
        Perform a forward pass through the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features).
            apply_sigmoid (bool): Whether to apply sigmoid activation
                to the output.

        Returns:
            torch.Tensor: Model output logits or probabilities.
        """
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.layer_norm(out)
        out = self.fc(out)

        if apply_sigmoid:
            return self.sigmoid(out)
        else:
            return out


def load_model_state_from_folder(
    folder_id: str,
    filename: str
):
    """
    Load a serialized PyTorch model state from a Dataiku folder.

    Args:
        folder_id (str): Dataiku folder identifier.
        filename (str): Name of the model state file.

    Returns:
        io.BytesIO: In-memory buffer containing the model state.

    Raises:
        Exception: If loading the model state fails.
    """
    logger.info("Loading Pytorch model state from folder...")
    try:
        folder = dataiku.Folder(folder_id)
        with folder.get_download_stream(filename) as f:
            buffer = io.BytesIO(f.read())
            buffer.seek(0)
            model_state = buffer

        logger.info("Pytorch model state loaded successfully.")
        return model_state

    except Exception as e:
        logger.error("Failed to load the Pytorch model state: %s", e, exc_info=True)
        raise


def create_model_and_load_state(
    model_filename_key: str,
    num_features: int
):
    """
    Create a model instance and load its trained state.

    The model type is inferred from the model filename and parameters
    are retrieved from project configuration.

    Args:
        model_filename_key (str): Configuration key pointing to the model filename.
        num_features (int): Number of input features.

    Returns:
        torch.nn.Module or None: Loaded model instance or None if model
        type is not recognized.
    """
    logger.info("Creating Pytorch model and loading dict...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_filename = cfg.get_project_variable(model_filename_key)
    folder_id_models = cfg.get_project_variable(cfg.SAVED_MODELS_FOLDER_ID_KEY)

    model_state = load_model_state_from_folder(
        folder_id_models,
        model_filename
    )

    if 'CNN' in model_filename:
        params = cfg.get_project_variable(cfg.BEST_CNN_MODEL_PARAMS_KEY)
        kernel_sizes_list = [params[k] for k in params if 'kernel' in k]

        model_args = {
            "input_channels": num_features,
            "num_filters": params.get('num_filters', 64),
            "kernel_sizes": kernel_sizes_list,
            "num_layers": params.get('num_layers', 3),
            "dropout": params.get('dropout', 0.2),
            "use_pool": params.get('use_pool', True),
            "pool_type": params.get('pool_type', "max"),
            "use_batchnorm": params.get('use_batchnorm', True),
            "activation": params.get('activation', "ReLU"),
            "leaky_slope": params.get('leaky_slope', 0.01)
        }

        model = CNNModel(**model_args)
        model.load_state_dict(torch.load(model_state, map_location=device))
        model.to(device)

        return model
    else:
        logger.warning('Unknown model type, skipping training model.')
        return None
