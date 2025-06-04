from collections import OrderedDict
from typing import Optional, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from src.model.utils.feed_forward_network import FeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 ff_dim: int,
                 dropout: float
                 ) -> None:
        """
        Initialize the decoder block.

        Args:
            num_heads (int): The number of attention heads.
            d_model (int): The dimension of the hidden layers.
            ff_dim (int): The size of the feed-forward network.
            dropout (float): The dropout probability for regularization.
        """
        super(DecoderBlock, self).__init__()

        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # First dropout layer
        self.dropout_1 = nn.Dropout(dropout)

        # First layer normalization
        self.ln_1 = nn.LayerNorm(normalized_shape=d_model)

        # Multi-head attention mechanism
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

        # Second dropout layer
        self.dropout_2 = nn.Dropout(dropout)

        # Second layer normalization
        self.ln_2 = nn.LayerNorm(normalized_shape=d_model)

        # The feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model=d_model, ff_dim=ff_dim, dropout=dropout)

        # Third dropout layer
        self.dropout_3 = nn.Dropout(dropout)

        # Third layer normalization
        self.ln_3 = nn.LayerNorm(normalized_shape=d_model)

    @staticmethod
    def add_embedding(x: Tensor, emb: Optional[Union[Tensor, nn.Embedding]] = None):
        """
        Adds embeddings to an input tensor.

        Args:
            x (Tensor): The input tensor of shape (seq_length, batch_size, d_model).
            emb (Tensor): The embedding of shape (seq_length, batch_size, d_model).

        Returns:
            Tensor: The input tensor with embeddings added.
        """
        return x if emb is None else x + emb

    def forward(
            self,
            x: Tensor,
            enc_output: Tensor,
            pos_emb: Optional[Tensor] = None,
            query_emb: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the Decoder Block.

        Args:
            x (Tensor): The input tensor of shape (num_queries, batch_size, d_model).
            enc_output (Tensor): The encoder output tensor of shape (seq_len, batch_size, d_model).
            pos_emb (Tensor): The positional embedding of shape (seq_length, batch_size, d_model).
            query_emb (Tensor): The object query embeddings of shape (num_queries, batch_size, d_model).
            attention_mask (Tensor): The attention mask to apply.

        Returns:
            Tensor: The output tensor of shape (num_queries, batch_size, d_model).
        """
        # Add positional embedding
        q = k = self.add_embedding(x=x, emb=query_emb)

        # Calculate self-attention
        attn = self.self_attention(
            query=q,
            key=k,
            value=x,
            attn_mask=attention_mask
        )[0]

        # Concatenate input and attention output
        x = x + self.dropout_1(attn)

        # Apply layer normalization
        x = self.ln_1(x)

        attn_2 = self.multi_head_attention(
            query=self.add_embedding(x=x, emb=query_emb),
            key=self.add_embedding(x=enc_output, emb=pos_emb),
            value=enc_output,
            attn_mask=attention_mask
        )[0]

        # Concatenate input and attention output
        x = x + self.dropout_2(attn_2)

        # Apply layer normalization
        x = self.ln_2(x)

        # Pass through the feed-forward network block
        ff_output = self.feed_forward(x)

        # Concatenate input and attention output
        x = x + self.dropout_3(ff_output)

        # Apply layer normalization
        output = self.ln_3(x)

        return output


class Decoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 num_queries: int,
                 d_model: int,
                 ff_dim: int,
                 dropout: float,
                 auxiliary_losses: bool
                 ) -> None:
        """
        Initialize the decoder.

        Args:
            num_layers (int): The Number of decoder layers.
            num_heads (int): The Number of attention heads.
            num_queries (int): The number of object queries.
            d_model (int): The dimension of the hidden layers.
            ff_dim (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability for regularization.
            auxiliary_losses (bool): Whether to return outputs from all decoder layers for auxiliary losses.
        """
        super(Decoder, self).__init__()

        # Whether to return the output of each decoder layer or not
        self.auxiliary_losses = auxiliary_losses

        # Object queries
        self.object_queries = nn.Embedding(num_queries, d_model)

        # Initialize the layers of the decoder
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f'decoder_layer_{i}'] = DecoderBlock(
                num_heads=num_heads,
                d_model=d_model,
                ff_dim=ff_dim,
                dropout=dropout
            )
        self.layers = nn.Sequential(layers)

        # Final layer normalization
        self.ln = nn.LayerNorm(normalized_shape=d_model)

    def forward(
            self,
            encoder_output: Tensor,
            pos_emb: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None
    ) -> List[Tensor]:
        """
        Forward pass through all encoder layers.

        Args:
            encoder_output (Tensor): The output from the encoder of shape (seq_length, batch_size, d_model).
            pos_emb (Tensor): The positional embedding of shape (seq_length, batch_size, d_model).
            attention_mask (Tensor): The attention mask to apply.

        Returns:
            Tensor: The output tensors of shape (num_queries, batch_size, d_model).
        """
        # Get batch size
        batch_size = encoder_output.shape[1]

        # Object queries of shape (num_queries, batch_size, d_model)
        query_emb = self.object_queries.weight.unsqueeze(1).repeat(1, batch_size, 1)
        x = torch.zeros_like(query_emb)

        # Pass the input through the decoder layers
        outputs: List[Tensor] = list()
        for layer in self.layers:
            # Pass through the current decoder layer
            x = layer(x, enc_output=encoder_output, pos_emb=pos_emb, query_emb=query_emb, attention_mask=attention_mask)

            # Store the output of each layer if using auxiliary losses
            if self.auxiliary_losses:
                # Apply layer normalization to the output of each decoder layer
                outputs.append(self.ln(x))

        # Return the output of each decoder layer during the training
        if self.auxiliary_losses:
            return outputs

        # Apply layer normalization to the output of the decoder
        outputs = [self.ln(x)]

        return outputs
