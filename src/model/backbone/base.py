import torch
import torch.nn as nn
from torch import Tensor


class Backbone(nn.Module):
    def __init__(self, pretrained_weights: str = None, frozen_layers: bool = False, **kwargs) -> None:
        """
        Initializes the base embedding model.

        Args:
            pretrained_weights (str): Path to the pretrained weights.
            frozen_layers (bool): Whether the layers should be frozen or not.
            **kwargs: Additional arguments specific to the model.
        """
        super(Backbone, self).__init__()

        # Initialize the specific model
        self.backbone_model = self._initialize_model(**kwargs)

        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self.backbone_model.load_state_dict(torch.load(pretrained_weights, weights_only=True), strict=False)

        # Replace the last layer with a new one
        self._replace_last_layer()

        # Frozen backbone
        if frozen_layers:
            self._freeze_layers(self.backbone_model)

    def _freeze_layers(self, model: nn.Module):
        for name, child in model.named_children():
            # Freeze params
            for param in child.parameters():
                param.requires_grad = False

            self._freeze_layers(child)

    def _initialize_model(self, **kwargs) -> nn.Module:
        """
        Initializes the specific embedding model

        Args:
            **kwargs: Additional arguments specific to the model.

        Returns:
            nn.Module: The initialized model.
        """
        raise NotImplementedError("Subclasses must implement a '_initialize_model' method.")

    def _replace_last_layer(self) -> None:
        """
        Replaces the last layer of the model.
        """
        raise NotImplementedError("Subclasses must implement a '_replace_last_layer' method.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the backbone model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Pass the input tensor through the backbone model
        output = self.backbone_model(x)

        return output
