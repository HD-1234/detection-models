from collections import OrderedDict

import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, mlp_dim: int, output_dim: int, num_layers: int) -> None:
        """
        Initializes the MLP.

        Args:
            input_dim (int): The input dimension.
            mlp_dim (int): The dimension of the feed-forward network.
            output_dim (int): The output dimension.
            num_layers (int): The number of Linear layers.
        """
        super(MLP, self).__init__()

        # Calculate the total numbers of layers
        self.total_num_layers = (num_layers * 2) - 1

        # Define the layers of the MLP block using an OrderedDict
        layers = dict()
        for i in range(self.total_num_layers):
            if i % 2 == 0:
                layers[f'{i}'] = nn.Linear(
                    input_dim if i == 0 else mlp_dim, mlp_dim if i != self.total_num_layers - 1 else output_dim
                )
                nn.init.xavier_uniform_(layers[f'{i}'].weight)
            else:
                layers[f'{i}'] = nn.ReLU()

        self._modules = OrderedDict(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Iterate through the layers and apply them sequentially
        for i in range(self.total_num_layers):
            x = self._modules[str(i)](x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        """
        Initializes position-wise fully connected feed-forward network.

        Args:
            d_model (int): The dimension of the model.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed-forward network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
