from typing import Dict

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torch.optim as optim


class Scheduler:
    def __init__(
            self,
            model: nn.Module,
            device: str,
            loss_fn: nn.Module,
            optimizer: optim.Optimizer,
            training_loader: DataLoader,
            validation_loader: DataLoader
    ) -> None:
        """
        Initializes the scheduler.

        Args:
            model (nn.Module): The model to train.
            device (str): The device to use for training.
            loss_fn (nn.Module): The loss function to use.
            optimizer (optim.Optimizer): The optimizer to use.
            training_loader (DataLoader): The training data loader.
            validation_loader (DataLoader): The validation data loader.
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Train/Val loader
        self.training_loader = training_loader
        self.validation_loader = validation_loader

    def train_one_epoch(self, lr_scheduler: LRScheduler, max_norm: float = 0.1) -> Dict[str, Tensor]:
        """
        Trains the model for one epoch.

        Args:
            lr_scheduler (LRScheduler): The learn rate scheduler.
            max_norm (float): The maximum norm of the gradients.

        Returns:
            Dict[str, Tensor]: A dict with the accumulated losses.
        """
        # Set to train mode
        self.model.train()

        # Store all the losses
        train_loss = dict(
            loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            cls_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            bbox_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            giou_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
        )

        for ind, batch in enumerate(self.training_loader):
            # Unpack the current batch
            data = batch["image_tensor"].to(self.device)
            labels = batch["gt_tensor"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # Make predictions for this batch
            predictions = self.model(data)

            # Compute the loss and its gradients
            losses = self.loss_fn(predictions, labels, mask)
            losses["loss"].backward()

            # Clip the gradient norm
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            # Update the model parameters by applying the computed gradients
            self.optimizer.step()

            # Update the losses
            train_loss["loss"] += losses["loss"].item()
            train_loss["cls_loss"] += losses["cls_loss"].item()
            train_loss["bbox_loss"] += losses["bbox_loss"].item()
            train_loss["giou_loss"] += losses["giou_loss"].item()

        # Adjust the learning rate
        lr_scheduler.step()

        return train_loss

    def evaluate(self) -> Dict[str, Tensor]:
        """
        Evaluates the model.

        Returns:
            Dict[str, Tensor]: The evaluation results.
        """
        # Set to eval mode
        self.model.eval()

        # Store all the losses
        val_loss = dict(
            loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            cls_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            bbox_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            giou_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
        )

        with torch.no_grad():
            for (ind, batch) in enumerate(self.validation_loader):
                # Unpack the batch
                data = batch["image_tensor"].to(self.device)
                labels = batch["gt_tensor"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Make predictions for this batch
                predictions = self.model(data)

                # Compute the loss for the current batch and add it to the total validation loss
                losses = self.loss_fn(predictions, labels, mask)

                # Update losses
                val_loss["loss"] += losses["loss"].item()
                val_loss["cls_loss"] += losses["cls_loss"].item()
                val_loss["bbox_loss"] += losses["bbox_loss"].item()
                val_loss["giou_loss"] += losses["giou_loss"].item()

        return losses
