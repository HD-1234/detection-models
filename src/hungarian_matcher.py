from typing import Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from torchvision.ops.boxes import generalized_box_iou, box_convert


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight: int = 1, bbox_weight: int = 5, giou_weight: int = 2):
        """
        Initialize the Hungarian matcher with cost weights.

        Args:
            class_weight (int): Weight for classification cost.
            bbox_weight (int): Weight for L1 bounding box cost.
            giou_weight (int): Weight for GIoU bounding box cost.
        """
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

        if any([w < 1 for w in [class_weight, bbox_weight, giou_weight]]):
            raise ValueError("All cost weights have to be greater or equal to 1")

    @torch.no_grad()
    def forward(
            self,
            predicted_labels: torch.Tensor,
            predicted_bboxes: torch.Tensor,
            ground_truth_labels: torch.Tensor,
            ground_truth_bboxes: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the optimal matching between predictions and ground truth.

        Args:
            predicted_labels (torch.Tensor): The predicted class probabilities.
            predicted_bboxes (torch.Tensor): The predicted bboxes.
            ground_truth_labels (torch.Tensor): The ground truth labels.
            ground_truth_bboxes (torch.Tensor): The ground truth bboxes.
            mask (torch.Tensor): A mask for the valid ground truth objects.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: The indices of the matched predictions and the indices of the
                    matched ground truth elements.
        """
        # Get the device, the batch size and the number of object queries
        device = ground_truth_labels.device
        bs, num_queries = ground_truth_labels.shape[:2]

        # Mask out additional elements in the ground truth
        mask = mask[:, :, :1].squeeze(dim=-1)

        # Get the number of elements in the ground truth
        number_gt_elements = [(m == 1).sum().item() for m in mask]

        # Apply the mask to the relevant ground truth elements
        target_labels = ground_truth_labels[mask, ...].squeeze(dim=-1)
        target_bboxes = ground_truth_bboxes[mask, ...]

        # Compute the classification cost
        cost_class = -predicted_labels[:, target_labels]

        # Compute the L1 cost
        cost_bbox = torch.cdist(predicted_bboxes, target_bboxes, p=1)

        # Compute the GIoU cost
        cost_giou = -generalized_box_iou(
            box_convert(predicted_bboxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_bboxes, in_fmt="cxcywh", out_fmt="xyxy")
        )

        # Combine the individual costs into the final cost matrix
        cost_matrix = self.bbox_weight * cost_bbox + self.class_weight * cost_class + self.giou_weight * cost_giou

        # Reshape the cost matrix to BatchSize x ObjectQueries x GroundTruthObjects and move everything to CPU
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        # Calculate the indices with the lowest cost for each element in the batch individually
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(number_gt_elements, -1))]

        # Convert the indices into a tensor and add batch offsets to create global indices
        prediction_indices = torch.cat(
            [
                torch.tensor(ind[0], dtype=torch.int, device=device) + (i * num_queries)
                for i, ind in enumerate(indices)
            ]
        )
        target_indices = torch.cat(
            [
                torch.tensor(ind[1], dtype=torch.int, device=device) + (i * num_queries)
                for i, ind in enumerate(indices)
            ]
        )

        return prediction_indices, target_indices
