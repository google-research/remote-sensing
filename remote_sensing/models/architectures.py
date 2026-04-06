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

"""A collection of architectures for remote sensing tasks.

This module currently contains a UNet-style architecture for semantic
segmentation with ViT encoders.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
  """A basic convolutional block.

  This block consists of two KxK convolutions (where K is kernel_size), each
  followed by GroupNorm and ReLU.
  """

  def __init__(
      self, in_c: int, out_c: int, groups: int = 8, kernel_size: int = 3
  ):
    """Initializes the ConvBlock.

    Args:
        in_c: Number of input channels.
        out_c: Number of output channels.
        groups: Number of groups for GroupNorm.
        kernel_size: Size of the convolutional kernels.
    """
    super().__init__()

    if in_c <= 0 or out_c <= 0:
      raise ValueError("in_c and out_c must be positive integers.")
    if groups <= 0:
      raise ValueError("groups must be a positive integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
      raise ValueError("kernel_size must be a positive odd integer.")

    if out_c % groups != 0:
      raise ValueError("out_c must be divisible by groups.")

    self.block: nn.Sequential = nn.Sequential(
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(groups, out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_c,
            out_c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(groups, out_c),
        nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the convolutional block.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    return self.block(x)


class ViTUNetDecoder(nn.Module):
  """A UNet-style decoder for Vision Transformer features.

  This decoder takes a list of features from different encoder stages,
  progressively upsamples them, and combines them with skip connections
  to produce a dense prediction map.

  In a combined ViT + ViTUNetDecoder model the ViT patch size should be a
  power of 2 (i.e., 2^n), and the length of `decoder_channels` should usually
  be n + 1.
  """

  def __init__(
      self,
      encoder_dim: int,
      decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
      output_dims: int = 1,
  ):
    super().__init__()

    if not decoder_channels:
      raise ValueError("decoder_channels must not be empty.")
    if output_dims <= 0:
      raise ValueError("output_dims must be a positive integer.")

    self.center = ConvBlock(encoder_dim, decoder_channels[0])

    up_blocks = []
    conv_blocks = []
    for i in range(len(decoder_channels) - 1):
      up_blocks.append(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
      )
      conv_blocks.append(
          ConvBlock(decoder_channels[i] + encoder_dim, decoder_channels[i + 1])
      )

    self.up_blocks = nn.ModuleList(up_blocks)
    self.conv_blocks = nn.ModuleList(conv_blocks)

    self.refine = ConvBlock(decoder_channels[-1], decoder_channels[-1])
    self.pred_head = nn.Conv2d(
        decoder_channels[-1], output_dims, kernel_size=1
    )

  def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    """Forward pass for the decoder.

    Args:
        features: List of features from encoder layers. Expects N tensors of
          shape (B, C, H, W), where B is batch size, C is encoder dimension,
          and H, W are spatial dimensions. Features are expected to be ordered
          from shallowest to deepest layers.

    Returns:
        Segmentation map logits.
    """
    if len(features) != len(self.conv_blocks) + 1:
      raise ValueError(
          f"Expected {len(self.conv_blocks) + 1} features, but"
          f" got {len(features)}."
      )

    x = self.center(features[-1])

    for i, (up, conv) in enumerate(zip(self.up_blocks, self.conv_blocks)):
      x = up(x)
      skip_feat = features[-2 - i]
      skip_feat = F.interpolate(
          skip_feat, size=x.shape[2:], mode="bilinear", align_corners=False
      )
      x = torch.cat([x, skip_feat], dim=1)
      x = conv(x)

    x = self.refine(x)
    return self.pred_head(x)
