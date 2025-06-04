from typing import Tuple, Dict, Union, List

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops.boxes import box_convert

from src.hungarian_matcher import HungarianMatcher


class DETRLoss(Module):
    def __init__(
            self,
            device: str,
            num_classes: int,
            class_weight: int,
            bbox_weight: int,
            giou_weight: int,
            background_class_weight: int,
            background_class_index: int = 0
    ) -> None:
        """
        Initializes the DETR loss module.

        Args:
            device (str): The device to use for training.
            num_classes (int): The number of classes.
            class_weight (int): The weight coefficient for classification loss.
            bbox_weight (int): The weight coefficient for L1 box regression loss.
            giou_weight (int): The weight coefficient for GIoU box regression loss.
            background_class_weight (int): The weight coefficient for the background class.
            background_class_index (int): Index of the background class.
        """
        super(DETRLoss, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.background_class_index = background_class_index

        # Initialize the hungarian matcher
        self.matcher = HungarianMatcher(
            class_weight=self.class_weight,
            bbox_weight=self.bbox_weight,
            giou_weight=self.giou_weight
        )

        # Create weights for classification loss. The background class gets a special weight.
        cls_loss_weights = torch.ones(num_classes + 1, device=self.device)
        cls_loss_weights[background_class_index] = background_class_weight
        self.register_buffer('cls_loss_weights', cls_loss_weights)

    @staticmethod
    def calculate_bbox_loss(
            predicted_bboxes: torch.Tensor,
            gt_bboxes: torch.Tensor,
            predicted_indices: torch.Tensor,
            gt_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates L1 and GIoU loss for the matched bboxes.

        Args:
            predicted_bboxes (torch.Tensor): The predicted bboxes.
            gt_bboxes (torch.Tensor): The ground truth bboxes.
            predicted_indices (torch.Tensor): The indices of the matched predicted bboxes.
            gt_indices (torch.Tensor): The indices of the matched ground truth bboxes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: L1 loss and GIoU loss.
        """
        # Get the number of matched ground truth bboxes
        num_bboxes = gt_indices.size(dim=0)

        # Extract the matched predictions and the ground truth
        source_bboxes = predicted_bboxes[predicted_indices]
        target_bboxes = gt_bboxes[gt_indices]

        # Convert the bboxes to XYXY format
        source_bboxes = box_convert(source_bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        target_bboxes = box_convert(target_bboxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Calculate the L1 loss
        bbox_loss = F.l1_loss(
            source_bboxes,
            target_bboxes,
            reduction='none'
        )

        # Calculate the average L1 loss over all bboxes
        bbox_loss = bbox_loss.sum() / num_bboxes

        # Calculate the GIoU loss
        giou_loss = generalized_box_iou_loss(
            source_bboxes,
            target_bboxes
        )

        # Calculate the average GIoU loss over all bboxes
        giou_loss = giou_loss.sum() / num_bboxes

        return bbox_loss, giou_loss

    def calculate_classification_loss(
            self,
            predicted_logits: torch.Tensor,
            target_labels: torch.Tensor,
            predicted_indices: torch.Tensor,
            gt_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the classification loss.

        Args:
            predicted_logits (torch.Tensor): The predicted class logits.
            target_labels (torch.Tensor): The ground truth class labels.
            predicted_indices (torch.Tensor): The indices of the matched predictions.
            gt_indices (torch.Tensor): The indices of the matched ground truth labels.

        Returns:
            torch.Tensor: Weighted cross-entropy classification loss.
        """
        # Extract matched ground truth classes
        target_labels_matched = target_labels[gt_indices]

        # Create a tensor of the background class index and overwrite the labels at the specified indices to represent
        # the ground truth elements
        target_classes = torch.full(
            target_labels.shape,
            fill_value=self.background_class_index,
            dtype=torch.int,
            device=self.device
        )
        target_classes[predicted_indices] = target_labels_matched

        # Calculate the cross-entropy loss with class specific weights
        cls_loss = F.cross_entropy(predicted_logits, target_classes, self.cls_loss_weights)

        return cls_loss

    def calculate_losses(
            self,
            predictions: Dict[str, torch.Tensor],
            gt_labels: torch.Tensor,
            gt_bboxes: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the losses for one layer.

        Args:
            predictions (Dict[str, torch.Tensor]): A dict with the predicted labels and bboxes.
            gt_labels (torch.Tensor): The ground truth labels.
            gt_bboxes (torch.Tensor): The ground truth bboxes.
            mask (torch.Tensor): A mask for the valid ground truth objects.

        Returns:
            Tuple: The combined loss, the classification loss, the L1 loss and the GIoU loss.
        """
        # Get the labels
        predicted_labels = predictions["labels"].flatten(0, 1).softmax(-1)

        # (batch_size, num_queries, 4) -> (batch_size * num_queries, 4)
        predicted_bboxes = predictions["bboxes"].reshape(-1, 4)

        # Split ground truth
        predicted_indices, gt_indices = self.matcher(
            predicted_labels=predicted_labels,
            predicted_bboxes=predicted_bboxes,
            ground_truth_labels=gt_labels,
            ground_truth_bboxes=gt_bboxes,
            mask=mask
        )

        # Flatten the ground truth labels
        gt_labels = gt_labels.flatten()

        # (batch_size, num_queries, 4) -> (batch_size * num_queries, 4)
        gt_bboxes = gt_bboxes.reshape(-1, 4)
        predicted_logits = predictions["labels"].reshape(-1, self.num_classes + 1)

        # Calculate the classification loss
        cls_loss = self.calculate_classification_loss(
            predicted_logits=predicted_logits,
            target_labels=gt_labels,
            predicted_indices=predicted_indices,
            gt_indices=gt_indices
        )

        # Calculate the L1 and GIoU loss for the matched bboxes
        bbox_loss, giou_loss = self.calculate_bbox_loss(
            predicted_bboxes=predicted_bboxes,
            gt_bboxes=gt_bboxes,
            predicted_indices=predicted_indices,
            gt_indices=gt_indices
        )

        # Apply the weights and calculate the combined loss
        cls_loss = self.class_weight * cls_loss
        bbox_loss = self.bbox_weight * bbox_loss
        giou_loss = self.giou_weight * giou_loss
        total_loss = cls_loss + bbox_loss + giou_loss

        return total_loss, cls_loss, bbox_loss, giou_loss

    def forward(
            self,
            predictions: Dict[str, Union[torch.Tensor, list]],
            ground_truth: torch.Tensor,
            mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the total and the individual losses for each layer.

        Args:
            predictions (Dict[str, Union[torch.Tensor, list]]): A dict with the predicted labels and bboxes.
            ground_truth (torch.Tensor): A tensor containing the ground truth labels and the ground truth bboxes.
            mask (torch.Tensor): A mask for the valid ground truth objects.

        Returns:
            Dict[str, torch.Tensor]: A dict containing the total loss, the classification loss, the L1 loss and the
            GIoU loss.
        """
        # Extract labels and bboxes from the ground truth tensor
        gt_labels = ground_truth[:, :, :1].to(torch.int)
        gt_bboxes = ground_truth[:, :, 1:]

        # Prepare the predictions for the loss calculation
        predictions_unpacked: List[dict] = list()
        if "auxiliary_outputs" in predictions:
            predictions_unpacked.extend(predictions["auxiliary_outputs"])
        predictions_unpacked.append(dict(labels=predictions["labels"], bboxes=predictions["bboxes"]))

        # Initialize the loss dict
        losses = dict(
            loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            cls_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            bbox_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
            giou_loss=torch.tensor(0.0, dtype=torch.float, device=self.device),
        )

        # Compute the corresponding losses for each layer
        for pred in predictions_unpacked:
            loss, cls_loss, bbox_loss, giou_loss = self.calculate_losses(
                predictions=pred,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                mask=mask
            )

            # Update losses
            losses["loss"] += loss
            losses["cls_loss"] += cls_loss
            losses["bbox_loss"] += bbox_loss
            losses["giou_loss"] += giou_loss

        return losses
