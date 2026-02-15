"""
trainer module

This module provides functions for training PyTorch models for binary classification
tasks on sequential data. 

It includes:
- train_loop: The main training loop with support for BCE, weighted BCE, and FocalLoss.
- Dynamic adjustment of sampler weights or FocalLoss alpha based on class recall.
- Early stopping based on validation loss.
- Checkpointing the best model during training to a Dataiku folder.
- compute_new_weights: Adjust sampler weights dynamically.
- train_model_with_params: High-level function to instantiate a model, configure training
  parameters, and run the training loop.

The module is designed for industrial additive manufacturing process data,
handling per-part and per-subpart sequences with imbalanced classes.
"""

import os
import io
from typing import Dict, Any
import logging
from datetime import datetime
import __main__


import dataiku
from dataikuapi.utils import DataikuException
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, fbeta_score

from losses_and_metrics import FocalLoss
from model_handler import CNNModel #LSTMModel, GRUModel,
import project_globals as cfg

logger = logging.getLogger(__name__)

logger.info("---#### Starting training model ####---")

def train_loop(
    model,
    train_loader,
    train_loader_name,
    val_loader,
    device,
    learning_rate,
    batch_size,
    epochs=10,
    loss_over_train_chart=False,
    loss_type="default",
    weight=5,
    patience=5,
    dynamic_adjust=False,
    model_checkpoint=None
) -> Dict:
    """
    Train a PyTorch model with optional dynamic adjustment and early stopping.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        train_loader_name (str): Identifier for the training DataLoader type.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run training on (CPU or GPU).
        learning_rate (float): Learning rate for optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs for training.
        loss_over_train_chart (bool): Whether to plot train/val loss and recall.
        loss_type (str): Type of loss function ('default', 'weighted', 'focal').
        weight (float): Positive class weight for weighted BCE loss.
        patience (int): Early stopping patience (number of epochs with no val loss improvement).
        dynamic_adjust (bool): Whether to dynamically adjust weights or FocalLoss alpha.
        model_checkpoint (dict or None): Parameters for saving model checkpoints
            including folder_id and timestamp.

    Returns:
        dict: Dictionary of best metrics during training, including epoch, F2 score,
              precision, recall, F1 for each class, and validation loss.
    """

    last_checkpoint_filename = None
    model_name = str(type(model)).split("'")[1].split(".")[-1]
    logger.info(
        "Starting training for model: %s | Loss function: %s | DataLoader: %s | Batch size: %d",
        model_name, loss_type, train_loader_name, batch_size
    )

    # Inicjalizacja funkcji strat (loss) – dostosowanie dla BCEWithLogitsLoss lub FocalLoss
    if loss_type == "weighted":
        pos_weight = torch.tensor([weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=2, alpha=0.5, dynamic_adjust=dynamic_adjust) # MODYFIKACJA GAMMA
    else:  # default
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    # Zmienna do przechowywania najlepszego zestawu metryk
    best_metrics = {
        'epoch': 0,
        'f2_ng': -1.0,
        'recall_ng': 0.0,
        'precision_ng': 0.0,
        'f1_ng': 0.0,
        'recall_ok': 0.0,
        'precision_ok': 0.0,
        'f1_ok': 0.0,
        'val_loss': float('inf')
    }

    train_losses = []
    val_losses = []
    recalls_class_1 = []

    actual_epochs = 0

    for epoch in range(epochs):
        actual_epochs += 1
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch, apply_sigmoid=False)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        predictions = []
        y_val_tensor = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch, apply_sigmoid=False)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(1)
                loss = criterion(outputs, y_batch)
                predicted_labels = (torch.sigmoid(outputs) > 0.5).long()
                predictions.extend(predicted_labels.tolist())
                y_val_tensor.extend(y_batch.tolist())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        report = classification_report(y_val_tensor, predictions, zero_division=0, output_dict=True)
        f2_ng = fbeta_score(y_val_tensor, predictions, beta=2.0, zero_division=0)
        recalls_class_1.append(report.get("1.0", {}).get("recall", 0))

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f | Val Loss: %.4f | "
            "Recall (NG): %.4f | Precision (NG): %.4f | F2(beta=2) (NG): %.4f",
            epoch + 1, epochs, train_loss, val_loss,
            report.get("1.0", {}).get("recall", 0),
            report.get("1.0", {}).get("precision", 0),
            f2_ng
        )

        if ((report.get("1.0", {}).get("precision", 0) == 0.0) or
            (report.get("0.0", {}).get("precision", 0) == 0.0)):
            logger.warning("Precision is zero for at least one class. Stopping training!")
            best_metrics["f2_ng"] = 0 # Set metric to 0 to ensure it won't be selected by optuna
            return best_metrics

        if epoch >= 5 and epoch % 2 == 0:
            # Dynamicznie dostosuj probability wyboru klasy na podstawie wyników recall - DataLoader
            if dynamic_adjust and train_loader_name == "WeightedSampler":
                sampler_weights = train_loader.sampler.weights
                # Jeśli przez 5 ostatnich epok recall klasy 1 jest niski, zwiększ wagę klasy 1
                if sum(recall > 0.9 for recall in recalls_class_1[-10:]) == 0:
                    new_weights = compute_new_weights(sampler_weights, adjust_factor_class_1 = 1.1)
                    train_loader.sampler.weights = new_weights.to(dtype=sampler_weights.dtype,
                                                                   device=sampler_weights.device)
                # Jeśli przez 5 ostatnich epok recall klasy 1 jest wysoki, zwiększ wagę klasy 0
                elif sum(recall > 0.9 for recall in recalls_class_1[-5:]) == 5:
                    new_weights = compute_new_weights(sampler_weights, adjust_factor_class_1 = 0.9)
                    train_loader.sampler.weights = new_weights.to(dtype=sampler_weights.dtype,
                                                                   device=sampler_weights.device)

            # Dynamicznie dostosuj probability wyboru klasy na podstawie wyników recall - FocaLoss
            if dynamic_adjust and loss_type == "focal":
                # Jeśli 5 ostatnich epok recall klasy 1 jest niski, zwiększ wagę klasy 1 o 0.025
                if sum(recall > 0.9 for recall in recalls_class_1[-5:]) == 0:
                    new_alpha = min(criterion.alpha + 0.025, 0.9)
                    criterion.update_alpha(new_alpha)

                # Jeśli 5 ostatnich epok recall klasy 1 wysoki, zmniejsz wagę klasy 1 o 0.025
                elif sum(recall > 0.9 for recall in recalls_class_1[-5:]) == 5:
                    new_alpha = max(0.1, criterion.alpha - 0.025)
                    criterion.update_alpha(new_alpha)

        if (f2_ng > best_metrics['f2_ng'] or
            (f2_ng == best_metrics['f2_ng'] and val_loss < best_metrics['val_loss'])):
            best_metrics.update({
                'epoch': actual_epochs,
                'f2_ng': round(f2_ng, 3),
                'recall_ng': round(report.get("1.0", {}).get("recall", 0), 3),
                'precision_ng': round(report.get("1.0", {}).get("precision", 0), 3),
                'f1_ng': round(report.get("1.0", {}).get("f1-score", 0), 3),
                'recall_ok': round(report.get("0.0", {}).get("recall", 0), 3),
                'precision_ok': round(report.get("0.0", {}).get("precision", 0), 3),
                'f1_ok': round(report.get("0.0", {}).get("f1-score", 0), 3),
                'val_loss': round(val_loss, 3)
            })

            # Zapis modelu z epoki dla której metryki były najlepsze podczas treningu
            if model_checkpoint and model_checkpoint.get("active"):
                folder_id = model_checkpoint.get('folder_id')
                checkpoint_ts = model_checkpoint.get("timestamp")

                new_checkpoint_filename = (
                    f"best_{model_name}_epoch_{actual_epochs}_{checkpoint_ts}.pth"
                )

                try:
                    folder = dataiku.Folder(folder_id)
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    with folder.get_writer(new_checkpoint_filename) as w:
                        w.write(buffer.read())

                    logger.info("New best model checkpoint saved for epoch %d.", actual_epochs)
                    cfg.modify_project_variable(
                        cfg.BEST_MODEL_FILENAME_KEY,
                        new_checkpoint_filename
                    )

                    if last_checkpoint_filename:
                        try:
                            folder.delete_path(last_checkpoint_filename)
                            logger.info("Deleted old checkpoint: '%s'.", last_checkpoint_filename)
                        except (OSError, IOError) as e:
                            logger.warning("Could not delete old checkpoint '%s'. Reason: %s",
                                           last_checkpoint_filename, e, exc_info=True)
                        except Exception as e:
                            logger.exception("Unexpected error during checkpoint save: %s", e)
                            raise

                    last_checkpoint_filename = new_checkpoint_filename

                except (OSError, IOError, DataikuException) as e:
                    logger.error("Failed to save checkpoint '%s'. Reason: %s",
                                new_checkpoint_filename, e, exc_info=True)
                except Exception as e:
                    logger.exception("Unexpected error during checkpoint save: %s", e)
                    raise

        if patience_counter >= patience:
            logger.info("Early stopping triggered: no progress val_loss for %d epochs.", patience)
            break

    if loss_over_train_chart:
        plt.figure(figsize=(10, 5))
        plt.subplot(2,1,1)
        plt.plot(range(1, actual_epochs+1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, actual_epochs+1), val_losses, label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.xticks(range(1, actual_epochs+1))
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(range(1, actual_epochs+1), recalls_class_1, label='Recall (NG)', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Recall class 1(NG)')
        plt.title('Recall over epochs')
        plt.xticks(range(1, actual_epochs+1))
        plt.tight_layout()
        plt.show()

    return best_metrics

def compute_new_weights(actual_weights, adjust_factor_class_1):
    """
    Dynamically adjust sampler weights for imbalanced class sampling.

    Args:
        actual_weights (torch.Tensor): Current sample weights tensor.
        adjust_factor_class_1 (float): Multiplicative factor for class 1 weights.

    Returns:
        torch.Tensor: Updated sample weights tensor.
    """
    adjust_factor_class_0 = 2 - adjust_factor_class_1

    new_weights = actual_weights.clone()

    threshold = actual_weights.mean()

    mask_class_1 = actual_weights > threshold
    mask_class_0 = ~mask_class_1

    new_weights[mask_class_1] *= adjust_factor_class_1
    new_weights[mask_class_0] *= adjust_factor_class_0

    unique_vals = torch.unique(new_weights).cpu().numpy()
    print(f"Nowe wagi, klasa 0: {unique_vals[0]}, klasa 1: {unique_vals[1]}")
    return new_weights


def train_model_with_params(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    High-level function to instantiate and train a PyTorch model using given parameters.

    Args:
        model_type (str): Type of model to train ('CNN', others not implemented).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        params (dict): Dictionary containing hyperparameters, e.g.,
            - model_type
            - learning_rate
            - loss_type
            - weight
            - batch_size
            - num_filters
            - kernel sizes
            - dropout
            - activation
            - use_pool
            - pool_type
            - use_batchnorm
            - leaky_slope

    Returns:
        dict: Dictionary of best metrics achieved during training.
    """
    execution_mode = cfg.EXECUTION_MODE
    logger.info("--- Starting model %s train in mode: %s ---", model_type, execution_mode)

    num_features = next(iter(train_loader))[0].shape[-1]
    patience = (cfg.get_project_variable(cfg.TRAINING_PARAMS_KEY)).get('patience', 5)
    final_epochs = (cfg.get_project_variable(cfg.TRAINING_PARAMS_KEY)).get('epochs', 100)

    model_type = params.get('model_type', None)
    learning_rate = params.get('learning_rate', 0.001)
    loss_type = params.get('loss_type', 'default')
    weight = params.get('weight', 10)

    new_batch_size = params.get('batch_size', 64)
    train_loader = DataLoader(
        dataset=train_loader.dataset,
        batch_size=new_batch_size,
        sampler=train_loader.sampler,
        shuffle=(train_loader.sampler is None)
    )
    train_loader_name = 'WeightedRandomSampler'
    val_loader = DataLoader(
        dataset=val_loader.dataset,
        batch_size=new_batch_size * 2,
        shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    if model_type == 'CNN':
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
        model = CNNModel(**model_args).to(device)
    else:
        logger.warning('Unknown model type, skipping training model.')
        return None

    model_folder_id = cfg.get_project_variable(cfg.SAVED_MODELS_FOLDER_ID_KEY)

    training_ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    model_checkpoint = {"active": True,
                        "timestamp": training_ts,
                        "folder_id": model_folder_id}

    best_metrics = train_loop(
        model,
        train_loader,
        train_loader_name,
        val_loader,
        device,
        learning_rate,
        new_batch_size,
        epochs=final_epochs,
        loss_over_train_chart=False,
        loss_type=loss_type,
        weight=weight,
        patience=patience,
        dynamic_adjust=False,
        model_checkpoint=model_checkpoint)

    return best_metrics
