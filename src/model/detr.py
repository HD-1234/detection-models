from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.model.neck.decoder import Decoder
from src.model.neck.encoder import Encoder
from src.model.utils.feed_forward_network import MLP
from src.model.utils.positional_encoding import PositionEmbeddingSine
from src.modelloader import ModelLoader


class DETR(nn.Module):
    def __init__(
            self,
            backbone_model: str,
            backbone_weights: str,
            freeze_backbone: bool,
            freeze_batch_norm: bool,
            img_size: int,
            num_classes: int,
            num_heads: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_queries: int,
            hidden_dim: int,
            ff_dim: int,
            dropout: float,
            auxiliary_losses: bool = True
    ) -> None:
        """
        Initialize the DETR model.

        Args:
            backbone_model (str): The name of the backbone to use.
            backbone_weights (str): Path to the pretrained weights.
            freeze_backbone (bool): Whether the backbone should be frozen or not.
            freeze_batch_norm (bool): Whether to use BatchNorm2d layers where the batch statistics and
                the affine parameters are fixed or not.
            img_size (int): The size of the input image.
            num_classes (int): Number of object classes.
            num_heads (int): The Number of attention heads.
            num_encoder_layers (int): The Number of encoder layers.
            num_decoder_layers (int): The Number of decoder layers.
            num_queries (int): The number of object queries.
            hidden_dim (int): The dimension of the hidden layers.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability for regularization.
            auxiliary_losses (bool): Whether to return outputs from all decoder layers for auxiliary losses.
        """
        super(DETR, self).__init__()

        # Backbone
        self.backbone = ModelLoader(
            backbone=backbone_model,
            backbone_weights=backbone_weights,
            image_size=img_size,
            frozen_layers=freeze_backbone,
            frozen_batch_norm=freeze_batch_norm
        ).load_model()

        # Convolutional layer
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Validate the hidden dimension
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim of '{hidden_dim}' is not an even number")

        # Define number of positional features
        num_pos_features = hidden_dim // 2

        # Initialize the positional embeddings
        self.position_embedding = PositionEmbeddingSine(num_pos_features=num_pos_features)

        # Encoder
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_model=hidden_dim,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # Decoder
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            num_queries=num_queries,
            d_model=hidden_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            auxiliary_losses=auxiliary_losses
        )

        # Parameter Initialization
        self._reset_parameters()

        # Output heads
        self.classification_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bounding_boxes_head = MLP(input_dim=hidden_dim, mlp_dim=hidden_dim, output_dim=4, num_layers=3)

    def _reset_parameters(self):
        """
        Initialize model parameters using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Dict[str, Union[torch.Tensor, list]]:
        """
        Forward pass through to the DETR model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Dict[str, Union[Tensor, List]]: A dict containing labels, bboxes and the auxiliary outputs.
        """
        # Backbone feature extraction
        x = self.backbone(x)

        # Project backbone features to hidden dimension
        x = self.conv(x)

        # Positional Encoding
        pos_emb = self.position_embedding(x)

        # (batch_size, d_model, height, width) -> (batch_size, d_model, seq_length)
        x = x.flatten(2)
        pos_emb = pos_emb.flatten(2)

        # (batch_size, d_model, seq_length) -> (seq_length, batch_size, d_model)
        x = x.permute(2, 0, 1)
        pos_emb = pos_emb.permute(2, 0, 1)

        # Encoder output of shape (seq_length, batch_size, d_model
        encoder_output = self.encoder(x, pos_emb=pos_emb)

        # List of decoder outputs each of shape (num_queries, batch_size, d_model)
        decoder_output = self.decoder(encoder_output, pos_emb=pos_emb)

        outputs: Dict[str, Union[torch.Tensor, list]] = dict()
        for ind, d in enumerate(decoder_output):
            # Pass through the classification head
            class_outputs = self.classification_head(d)

            # Pass through the bbox head and apply sigmoid activation function
            bbox_outputs = self.bounding_boxes_head(d).sigmoid()

            # (num_queries, batch_size, num_classes) -> (batch_size, num_queries, num_classes)
            class_outputs = class_outputs.permute(1, 0, 2)

            # (num_queries, batch_size, 4) -> (batch_size, num_queries, 4)
            bbox_outputs = bbox_outputs.permute(1, 0, 2)

            # Add auxiliary outputs
            if ind + 1 < len(decoder_output):
                if "auxiliary_outputs" not in outputs:
                    outputs["auxiliary_outputs"] = list()
                outputs["auxiliary_outputs"].append(dict(labels=class_outputs, bboxes=bbox_outputs))
                continue

            # Store final output
            outputs["labels"] = class_outputs
            outputs["bboxes"] = bbox_outputs

        return outputs
