from collections import OrderedDict
from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.model.utils.feed_forward_network import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 ff_dim: int,
                 dropout: float
                 ) -> None:
        """
        Initialize the encoder block.

        Args:
            num_heads (int): The number of attention heads.
            d_model (int): The dimension of the hidden layers.
            ff_dim (int): The size of the feed-forward network.
            dropout (float): The dropout probability for regularization.
        """
        super(EncoderBlock, self).__init__()

        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # First dropout layer
        self.dropout_1 = nn.Dropout(dropout)

        # First layer normalization
        self.ln_1 = nn.LayerNorm(normalized_shape=d_model)

        # The feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model=d_model, ff_dim=ff_dim, dropout=dropout)

        # Second dropout layer
        self.dropout_2 = nn.Dropout(dropout)

        # Second layer normalization
        self.ln_2 = nn.LayerNorm(normalized_shape=d_model)

    @staticmethod
    def add_positional_embedding(x: Tensor, pos_emb: Optional[Tensor] = None):
        """
        Adds embeddings to an input tensor.

        Args:
            x (Tensor): The input tensor of shape (seq_length, batch_size, d_model).
            pos_emb (Tensor): The positional embedding of shape (seq_length, batch_size, d_model).

        Returns:
            Tensor: The input tensor with positional embeddings added.
        """
        return x if pos_emb is None else x + pos_emb

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Encoder Block.

        Args:
            x (Tensor): The input tensor of shape (seq_length, batch_size, d_model).
            pos_emb (Tensor): The positional embedding of shape (seq_length, batch_size, d_model).
            attention_mask (Tensor): The attention mask to apply.

        Returns:
            Tensor: The output tensor of shape (seq_length, batch_size, d_model).
        """
        # Add positional embedding to query and key
        q = k = self.add_positional_embedding(x=x, pos_emb=pos_emb)

        # Calculate self-attention
        attn = self.self_attention(query=q, key=k, value=x, attn_mask=attention_mask)[0]

        # Concatenate input and attention output and apply dropout
        x = x + self.dropout_1(attn)

        # Apply layer normalization
        x = self.ln_1(x)

        # Pass through the feed-forward network
        ff_output = self.feed_forward(x)

        # Concatenate input and attention output and apply dropout
        x = x + self.dropout_2(ff_output)

        # Apply layer normalization
        x = self.ln_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 ff_dim: int,
                 dropout: float
                 ) -> None:
        """
        Initialize the encoder.

        Args:
            num_layers (int): The Number of encoder layers.
            num_heads (int): The Number of attention heads.
            d_model (int): The dimension of the hidden layers.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability for regularization.
        """
        super(Encoder, self).__init__()

        # Initialize and stack the layers of the encoder
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f'encoder_layer_{i}'] = EncoderBlock(
                num_heads=num_heads,
                d_model=d_model,
                ff_dim=ff_dim,
                dropout=dropout
            )
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            x (Tensor): The input tensor of shape (seq_length, batch_size, d_model).
            pos_emb (Tensor): The positional embedding of shape (seq_length, batch_size, d_model).
            attention_mask (Tensor): The attention mask to apply.

        Returns:
            Tensor: The output tensor of shape (seq_length, batch_size, d_model).
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, attention_mask=attention_mask)

        return x
