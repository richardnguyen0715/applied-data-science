"""U-Net architecture for diffusion models."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, embedding_dim: int, max_period: int = 10000) -> None:
        """
        Initialize time embedding.

        Args:
            embedding_dim: Dimension of the embedding.
            max_period: Maximum period for sinusoidal encoding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode timestep as sinusoidal embedding.

        Args:
            t: Timestep tensor of shape (batch_size,).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=device)
            * (torch.log(torch.tensor(self.max_period, dtype=torch.float32))
               / half_dim)
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ClassEmbedding(nn.Module):
    """Class label embedding."""

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        """
        Initialize class embedding.

        Args:
            num_classes: Number of classes.
            embedding_dim: Dimension of the embedding.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Embed class labels.

        Args:
            y: Class labels of shape (batch_size,).

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.embedding(y)


class ResidualBlock(nn.Module):
    """Residual block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        cond_embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embedding_dim: Dimension of time embedding.
            cond_embedding_dim: Dimension of conditioning embedding.
            dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time and conditioning embedding projections
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.cond_proj = nn.Linear(cond_embedding_dim, out_channels)

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.
            time_emb: Time embedding.
            cond_emb: Conditioning embedding.

        Returns:
            Output tensor.
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Add time and conditioning embeddings
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = h + self.cond_proj(cond_emb)[:, :, None, None]

        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        """
        Initialize attention block.

        Args:
            channels: Number of channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)

        self.qkv = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Output tensor.
        """
        batch, channels, height, width = x.shape
        h = F.silu(self.norm(x))

        # Reshape to (batch, channels, height*width)
        h = h.reshape(batch, channels, height * width)

        # Compute QKV
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Compute attention
        q = q.transpose(1, 2)  # (batch, height*width, channels)
        k = k.transpose(1, 2)  # (batch, height*width, channels)
        v = v.transpose(1, 2)  # (batch, height*width, channels)

        # Reshape for multi-head attention
        batch_size = q.shape[0]
        q = q.reshape(batch_size * self.num_heads, -1, channels // self.num_heads)
        k = k.reshape(batch_size * self.num_heads, -1, channels // self.num_heads)
        v = v.reshape(batch_size * self.num_heads, -1, channels // self.num_heads)

        # Compute attention scores
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)

        # Apply attention to values
        out = attn @ v

        # Reshape back
        out = out.reshape(batch_size, -1, channels)
        out = out.transpose(1, 2)
        out = self.proj(out)
        out = out.reshape(batch, channels, height, width)

        return x + out


class UNet(nn.Module):
    """U-Net architecture for conditional diffusion."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        num_residual_blocks: int = 2,
        attention_resolutions: Tuple[int] = (8, 16),
        num_classes: int = 10,
        class_embed_dim: int = 128,
        time_embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize U-Net for conditional diffusion.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            model_channels: Base number of channels.
            num_residual_blocks: Number of residual blocks per level.
            attention_resolutions: Resolutions to apply attention.
            num_classes: Number of classes.
            class_embed_dim: Class embedding dimension.
            time_embedding_dim: Time embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_residual_blocks = num_residual_blocks
        self.time_embedding_dim = time_embedding_dim

        # Time embedding
        self.time_embedding = TimestepEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # Class embedding
        self.class_embedding = ClassEmbedding(num_classes, class_embed_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_embed_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # Combine time and class embeddings
        self.cond_embedding_proj = nn.Linear(
            time_embedding_dim + time_embedding_dim, time_embedding_dim
        )

        # Encoder
        self.initial_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        down_channels = [model_channels]
        current_channels = model_channels

        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        for _ in range(2):
            for _ in range(num_residual_blocks):
                self.down_blocks.append(
                    ResidualBlock(
                        current_channels,
                        model_channels * (2 ** len(down_channels)),
                        time_embedding_dim,
                        time_embedding_dim,
                        dropout,
                    )
                )
                current_channels = model_channels * (2 ** len(down_channels))

            down_channels.append(current_channels)
            self.down_convs.append(
                nn.Conv2d(current_channels, current_channels, kernel_size=4, stride=2, padding=1)
            )

        # Middle
        self.middle_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.middle_blocks.append(
                ResidualBlock(
                    current_channels,
                    current_channels,
                    time_embedding_dim,
                    time_embedding_dim,
                    dropout,
                )
            )

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(2):
            up_channels = down_channels[-(i + 2)]
            for _ in range(num_residual_blocks):
                self.up_blocks.append(
                    ResidualBlock(
                        current_channels + up_channels,
                        up_channels,
                        time_embedding_dim,
                        time_embedding_dim,
                        dropout,
                    )
                )
                current_channels = up_channels

            self.up_convs.append(
                nn.ConvTranspose2d(
                    current_channels, current_channels, kernel_size=4, stride=2, padding=1
                )
            )

        # Output
        self.final_norm = nn.GroupNorm(32, model_channels)
        self.final_conv = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
            t: Timestep indices of shape (batch,).
            y: Class labels of shape (batch,).

        Returns:
            Output tensor of shape (batch, out_channels, height, width).
        """
        # Embeddings
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)

        class_emb = self.class_embedding(y)
        class_emb = self.class_mlp(class_emb)

        # Combine embeddings
        cond_emb = torch.cat([time_emb, class_emb], dim=-1)
        cond_emb = self.cond_embedding_proj(cond_emb)

        # Initial convolution
        h = self.initial_conv(x)

        # Encoder with skip connections
        skips = [h]
        for block, conv in zip(self.down_blocks, self.down_convs):
            h = block(h, time_emb, cond_emb)
            skips.append(h)
            h = conv(h)

        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, time_emb, cond_emb)

        # Decoder
        up_block_idx = 0
        for conv in self.up_convs:
            h = conv(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for _ in range(self.num_residual_blocks):
                h = self.up_blocks[up_block_idx](h, time_emb, cond_emb)
                up_block_idx += 1

        # Output
        h = F.silu(self.final_norm(h))
        output = self.final_conv(h)

        return output
