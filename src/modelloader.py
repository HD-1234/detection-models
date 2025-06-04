from torch import nn
from src.model.backbone.resnet import *
from src.utils import write_log_message


class ModelLoader:
    def __init__(
            self,
            backbone: str,
            backbone_weights: str,
            image_size: int,
            frozen_layers: bool,
            frozen_batch_norm: bool,
    ) -> None:
        """
        Initializes the model loader.

        Args:
            backbone (str): The name of the backbone to use.
            backbone_weights (int): The pretrained weights of the backbone.
            image_size (int): The size of the input image.
            frozen_layers (bool): Whether the backbone should be frozen or not.
            frozen_batch_norm (bool): Whether to use BatchNorm2d layers where the batch statistics and
                the affine parameters are fixed or not.
        """
        self.backbone = backbone.lower()
        self.backbone_weights = backbone_weights
        self.image_size = image_size
        self.frozen_layers = frozen_layers
        self.frozen_batch_norm = frozen_batch_norm

    def load_model(self) -> nn.Module:
        """
        Loads the corresponding model.

        Returns:
            nn.Module: The actual model.
        """
        backbone_mapping = {
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'resnet152': ResNet152,
            'resnext50': ResNeXt50,
            'resnext101': ResNeXt101
        }

        if self.backbone not in backbone_mapping:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        # Model name to actual model
        backbone_type = backbone_mapping[self.backbone]

        if self.frozen_layers:
            write_log_message("Freezing backbone layers.")
            
        if self.frozen_batch_norm:
            write_log_message("Using 'FrozenBatchNorm2d' layers in the backbone.")

        # Initialize the model
        model = backbone_type(
            pretrained_weights=self.backbone_weights,
            frozen_layers=self.frozen_layers,
            frozen_batch_norm=self.frozen_batch_norm,
            image_size=self.image_size
        )

        return model
