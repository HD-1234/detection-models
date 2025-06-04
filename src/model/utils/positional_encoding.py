import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionEmbeddingSine(nn.Module):
    def __init__(
            self,
            num_pos_features: int,
            temperature: int = 10000,
            normalize: bool = True,
            scale: float = None
    ):
        """
        Initialize the PositionEmbeddingSine module.

        Args:
            num_pos_features (int, optional): The number of positional features to compute for each position.
            temperature (int, optional): The temperature for scaling the sine and cosine functions.
            normalize (bool, optional): Whether to normalize the positional embeddings.
            scale (float, optional): The scaling factor for the embeddings. If None, it defaults to 2 * math.pi.

        Source:
            https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/position_encoding.py#L12
        """
        super().__init__()

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        """
        Compute positional embeddings using sine and cosine functions for each position in the input tensor.

        Args:
            x (Tensor): The input tensor of shape (batch_size, channels, height, width).
            eps (float): A small epsilon to avoid division by zero.

        Returns:
            Tensor: Positional embeddings tensor of shape (batch_size, num_pos_features * 2, height, width).
        """
        batch_size, _, height, width = x.shape

        # Create a tensor with all positions being valid
        mask_tensor = torch.ones(batch_size, height, width, dtype=torch.bool, device=x.device)

        # Compute cumulative sums along width to get position embeddings
        embedding_x = mask_tensor.cumsum(dim=2)

        # Compute cumulative sums along height to get position embeddings
        embedding_y = mask_tensor.cumsum(dim=1)

        # Normalize the embeddings if required
        if self.normalize:
            embedding_x = embedding_x / (embedding_x[:, :, -1:] + eps) * self.scale
            embedding_y = embedding_y / (embedding_y[:, -1:, :] + eps) * self.scale

        # Compute the temperature scaling factor
        dim_t = torch.arange(self.num_pos_features, dtype=torch.float, device=x.device)
        dim_t = torch.exp((2 * (dim_t // 2)) * (-math.log(self.temperature) / self.num_pos_features))

        # Compute positional features using sine and cosine
        pos_x = embedding_x.unsqueeze(-1) * dim_t
        pos_y = embedding_y.unsqueeze(-1) * dim_t

        # Apply sine and cosine functions
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=-1).flatten(3)

        # Concatenate the positional features and permute dimensions
        output = torch.cat((pos_y, pos_x), 3).permute(0, 3, 1, 2)

        return output
