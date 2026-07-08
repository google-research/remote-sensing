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

import collections
import torch
from torch import nn
import torch.nn.functional as F


OrderedDict = collections.OrderedDict


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


class ViTBackbonePyramid(nn.Module):
  """A Vision Transformer (ViT) backbone with a single-feature spatial pyramid.

  The final ViT patch-token representation is reshaped into a spatial
  feature map. Additional feature levels are produced using interpolation
  and pooling for use with Torchvision detection models.
  """

  def __init__(
      self,
      encoder: nn.Module,
      image_size: int,
      out_channels: int = 256,
      freeze_encoder: bool = False,
  ):
    """Initializes the ViT backbone pyramid.

    Args:
        encoder: Pretrained Vision Transformer encoder module.
        image_size: Input image resolution.
        out_channels: Number of channels in each pyramid feature map.
        freeze_encoder: Whether to freeze the pretrained ViT encoder.
    """
    super().__init__()

    self.encoder: nn.Module = encoder

    self.image_size: int = image_size
    self.patch_size: int = self.encoder.config.patch_size
    self.hidden_size: int = self.encoder.config.hidden_size

    if self.patch_size not in (8, 16, 32):
      raise ValueError(
          f"Expected patch size in (8, 16, 32), got {self.patch_size}."
      )

    if self.image_size % self.patch_size != 0:
      raise ValueError(
          f"image_size={self.image_size} must be divisible by "
          f"patch_size={self.patch_size}."
      )

    self.projection: nn.Sequential = nn.Sequential(
        nn.Conv2d(
            self.hidden_size,
            out_channels,
            kernel_size=1,
        ),
        nn.GroupNorm(32, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.GroupNorm(32, out_channels),
        nn.ReLU(inplace=True),
    )

    # Required by Torchvision detection models.
    self.out_channels: int = out_channels

    if freeze_encoder:
      for parameter in self.encoder.parameters():
        parameter.requires_grad = False

  def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Extracts the multi-scale ViT feature pyramid.

    Args:
        images: Input tensor with shape (B, C, H, W).

    Returns:
        Ordered dictionary containing feature maps at strides 4, 8, 16, 32, and
        64 relative to the input image spatial resolution (H, W).
    """
    encoder_output = self.encoder(images)
    tokens = encoder_output.last_hidden_state

    batch_size, num_tokens, hidden_dim = tokens.shape
    # The input spatial dimensions (height, width) directly determine the
    # resolution of each pyramid level, where level P_k has spatial stride 2^k
    # (i.e. dimensions height // 2^k and width // 2^k relative to the image).
    height, width = images.shape[-2:]

    patch_height = height // self.patch_size
    patch_width = width // self.patch_size
    expected_tokens = patch_height * patch_width

    if num_tokens != expected_tokens:
      raise ValueError(
          f"Unexpected token count: received {num_tokens}, "
          f"expected {expected_tokens} "
          f"({patch_height} x {patch_width})."
      )

    features = tokens.reshape(
        batch_size,
        patch_height,
        patch_width,
        hidden_dim,
    )

    features = features.permute(0, 3, 1, 2).contiguous()
    features = self.projection(features)

    if self.patch_size == 16:
      # For patch_size=16, the base feature map has spatial stride 16 (P4).
      # We define P4 first, then derive P3 and P2 via 2x upsampling, and P5 and
      # P6 via 2x max pooling.
      p4 = features
      p3 = F.interpolate(
          p4, scale_factor=2, mode="bilinear", align_corners=False
      )
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)
    elif self.patch_size == 8:
      # For patch_size=8, the base feature map has spatial stride 8 (level P3).
      # We derive P2 via upsampling, and P4, P5, and P6 via max pooling.
      p3 = features
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p4 = F.max_pool2d(p3, kernel_size=2, stride=2)
      p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)
    else:  # self.patch_size == 32
      # For patch_size=32, the base feature map has spatial stride 32 (level
      # P5). We derive P4, P3, and P2 via upsampling, and P6 via max pooling.
      p5 = features
      p4 = F.interpolate(
          p5, scale_factor=2, mode="bilinear", align_corners=False
      )
      p3 = F.interpolate(
          p4, scale_factor=2, mode="bilinear", align_corners=False
      )
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)

    return OrderedDict({
        "0": p2,
        "1": p3,
        "2": p4,
        "3": p5,
        "4": p6,
    })
