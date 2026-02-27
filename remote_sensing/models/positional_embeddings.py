# Copyright 2026 The Earth AI Remote Sensing Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Positional embeddings for Earth AI Remote Sensing models."""

import torch
from torch import nn
import transformers
from transformers.models.vit import modeling_vit


class Sincos2dEmbeddings(nn.Module):
  """Computes sin/cos 2d positional embeddings based on PositionalEmbedding2D from the praxis library."""

  def __init__(self, config: transformers.ViTConfig) -> None:
    super().__init__()

    self.patch_embeddings = modeling_vit.ViTPatchEmbeddings(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

    self.patch_size = config.patch_size
    self.config = config

  def forward(
      self,
      pixel_values: torch.Tensor,
  ) -> torch.Tensor:
    unused_batch_size, unused_num_channels, height, width = pixel_values.shape
    embeddings = self.patch_embeddings(
        pixel_values, interpolate_pos_encoding=True
    )

    embedding_h = self.config.image_size // self.patch_size
    embedding_w = self.config.image_size // self.patch_size

    pos_embeddings = self.compute_sincos2d_embeddings(
        embedding_h,
        embedding_w,
        self.config.hidden_size,
    ).to(embeddings.device)

    # Always interpolate when tracing to ensure the exported model works for
    # dynamic input shapes.
    if (
        torch.jit.is_tracing()
        or self.config.image_size != height
        or self.config.image_size != width
    ):
      pos_embeddings = self.interpolate_pos_encoding(
          pos_embeddings, height // self.patch_size, width // self.patch_size
      )

    embeddings += pos_embeddings
    embeddings = self.dropout(embeddings)

    return embeddings

  def _compute_1d_embeddings(
      self, positions: torch.Tensor, hidden_dim: int, dtype=torch.float32
  ) -> torch.Tensor:
    half_hid = hidden_dim // 2
    freq_seq = torch.arange(half_hid, dtype=dtype)
    positions = positions.type(dtype)
    # the base 10000 is from the original sinusoidal positional embedding
    # formulation introduced in "attention is all you need" section 3.5.
    # https://arxiv.org/pdf/1706.03762.pdf
    inv_freq = 1.0 / (10000.0 ** (freq_seq / half_hid))
    positions = torch.einsum('S,D->SD', positions, inv_freq)
    sin = torch.sin(positions)
    cos = torch.cos(positions)
    return torch.concatenate([sin, cos], axis=-1)

  def _compute_2d_embeddings(
      self, h: int, w: int, hidden_dim: int
  ) -> torch.Tensor:
    dim = hidden_dim
    h_seq = torch.arange(-h // 2, h // 2, dtype=torch.float32)
    w_seq = torch.arange(-w // 2, w // 2, dtype=torch.float32)
    pos_emb_h = self._compute_1d_embeddings(
        h_seq, dim // 2, dtype=torch.float32
    )
    pos_emb_w = self._compute_1d_embeddings(
        w_seq, dim // 2, dtype=torch.float32
    )
    pos_emb_2d = torch.concatenate(
        [
            torch.tile(pos_emb_h[:, None, :], [1, w, 1]),
            torch.tile(pos_emb_w[None, :, :], [h, 1, 1]),
        ],
        axis=-1,
    )
    return pos_emb_2d

  def interpolate_pos_encoding(
      self, embeddings: torch.Tensor, embedding_h: int, embedding_w: int
  ) -> torch.Tensor:
    """Interpolates positional embeddings to a new image size."""
    pretrained_grid_size = self.config.image_size // self.patch_size
    pos_emb_2d = embeddings.reshape(
        1, pretrained_grid_size, pretrained_grid_size, self.config.hidden_size
    )  # [B, H, W, D]
    pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)  # [B, D, H, W]
    interp_emb = torch.nn.functional.interpolate(
        pos_emb_2d,
        size=(embedding_h, embedding_w),
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )
    interp_emb = interp_emb.permute(0, 2, 3, 1)  # [B, H', W', D]
    interp_emb = interp_emb.view(1, -1, self.config.hidden_size)
    return interp_emb

  def compute_sincos2d_embeddings(
      self, h: int, w: int, hidden_dim: int
  ) -> torch.Tensor:
    """Generates a Tensor of sinusoids with different frequencies.

    Args:
      h: Image height (in patches).
      w: Image width (in patches).
      hidden_dim: The embedding dimension.

    Returns:
      2D positional embedding Tensor of shape
        [1, p.num_prepend_cls_tokens + p.h * p.w, D].
    """
    pos_emb = self._compute_2d_embeddings(h=h, w=w, hidden_dim=hidden_dim)
    pos_emb = torch.reshape(pos_emb, (h * w, hidden_dim))
    pos_emb = torch.unsqueeze(pos_emb, dim=0)
    return pos_emb
