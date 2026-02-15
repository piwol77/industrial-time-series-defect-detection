"""
losses_and_metrics module.

This module contains custom loss functions and evaluation utilities
for binary classification models trained on industrial process data.
It provides a Focal Loss implementation for handling class imbalance,
as well as model evaluation and scoring logic including probability
outputs and optional reporting metrics.
"""

import os
import logging
import __main__

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

import project_globals as cfg
import torch_data_preparer as tdp
import model_handler as models


logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.

    This loss function extends binary cross-entropy by down-weighting
    well-classified examples and focusing training on hard samples.
    It supports both static and dynamically adjusted class weighting.
    """

    def __init__(self, gamma=2, alpha=0.5, reduction='mean', dynamic_adjust=False):
        """
        Initialize the FocalLoss module.

        Args:
            gamma (float): Focusing parameter that controls the strength
                of down-weighting easy examples.
            alpha (float): Class weighting factor for the positive class.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
            dynamic_adjust (bool): Whether to dynamically adjust alpha
                per sample based on target label.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.dynamic_adjust = dynamic_adjust

    def update_alpha(self, new_alpha):
        """
        Update the alpha weighting parameter during training.

        Args:
            new_alpha (float): New alpha value to be applied.
        """
        self.alpha = new_alpha
        print(f"Zaktualizowano alpha do {self.alpha}")

    def forward(self, inputs, targets):
        """
        Compute the focal loss value.

        Args:
            inputs (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Obliczamy standardową stratę cross-entropy dla binarnej klasyfikacji, inputs: logity
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)  # Prawdopodobieństwo prawidłowej klasy

        # Dynamiczne ustalanie wagi w zależności od etykiety próbki
        if self.dynamic_adjust:
            # Dla próbki target == 1, stosujemy self.alpha, gdzie target == 0, stosujemy 1
            alpha_tensor = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=targets.device),
                torch.tensor(1 - self.alpha, device=targets.device)
            )
        else:
            # Przy statycznej wadze, używamy podanego skalara
            alpha_tensor = self.alpha

        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    labels=None
):
    """
    Plot a confusion matrix for classification results.

    Args:
        true_labels (array-like): Ground truth labels.
        predicted_labels (array-like): Predicted class labels.
        labels (list, optional): List of label names or values to display.
    """
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate_model(
    test_loader,
    threshold=0.5
) -> pd.DataFrame:
    """
    Evaluate or score a trained model on test data.

    This function loads the best saved model and corresponding scaler,
    performs inference on the provided test data loader, and returns
    either detailed evaluation results or scoring output depending on
    execution mode.

    Args:
        test_loader (torch.utils.data.DataLoader): DataLoader containing
            test features, labels, and metadata.
        threshold (float): Decision threshold for binary classification.

    Returns:
        pd.DataFrame: DataFrame containing predictions, probabilities,
        and optionally true labels.
    """
    execution_mode = cfg.EXECUTION_MODE
    logger.info(
        "--- Starting Model Evaluation/Scoring (Mode: %s) ---",
        execution_mode
    )

    num_features = next(iter(test_loader))[0].shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.create_model_and_load_state(
        cfg.BEST_MODEL_FILENAME_KEY,
        num_features
    )

    scaler_filename = cfg.get_project_variable(cfg.LAST_SCALER_FILENAME_KEY)
    folder_id = cfg.get_project_variable(cfg.SAVED_MODELS_FOLDER_ID_KEY)
    scaler = tdp.load_scaler_from_folder(folder_id, scaler_filename)

    model.eval()
    test_results = []

    with torch.no_grad():
        for x_batch, y_batch, metadata_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()

            # If batch size = 1, below code is neccessary
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            predictions = (outputs > threshold).long()
            for idx in range(len(x_batch)):
                subpart_no_inv = round(
                    float(
                        x_batch[idx, :, 0].mean().item()
                        * scaler.scale_[0]
                        + scaler.mean_[0]
                    ),
                    0
                )

                test_results.append({
                    "machine": cfg.MACHINE_NAME_KEY,
                    "part_number": metadata_batch[idx].item(),
                    "subpart_no": subpart_no_inv,
                    "real_label": y_batch[idx].item(),
                    "predicted_label": predictions[idx].item(),
                    "prob_OK": (1 - outputs[idx].item()),
                    "prob_NG": outputs[idx].item()
                })

    results_df = pd.DataFrame(test_results)

    if 'score' in execution_mode:
        results_df = results_df.drop(columns='real_label')
        results_df = results_df.rename(columns={'predicted_label': 'label'})
        logger.info("--- Scoring completed. Returning results_df ---")
        return results_df

    true_labels = results_df['real_label']
    predicted_labels = results_df['predicted_label']

    report = classification_report(true_labels, predicted_labels, digits=2)
    logger.info("\n%s", report)

    # plot_confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    return results_df
