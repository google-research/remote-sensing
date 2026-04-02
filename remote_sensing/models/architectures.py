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
from typing import Any, Callable

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


class ViTSegmentationModel(nn.Module):
  """ViT-based segmentation model with a UNet-style decoder.

  The model extracts intermediate features from selected encoder layers and
  uses them in a decoder that progressively upsamples the representation to
  produce a segmentation mask. Intermediate encoder features are captured
  with forward hooks, which allow the model to access layer outputs during
  the forward pass without modifying the encoder itself.

  PyTorch forward hooks:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
  """

  def __init__(
      self,
      encoder: nn.Module,
      decoder: nn.Module,
      hook_layer_indices: tuple[int, ...] = (5, 11, 17, 23),
  ):
    """Initializes the ViTSegmentationModel.

    Args:
        encoder: The Vision Transformer encoder.
        decoder: The UNet-style decoder.
        hook_layer_indices: Indices of encoder layers from which to extract
          intermediate features using forward hooks. These features are used
          as skip connections in the decoder.
    """
    super().__init__()

    if not hook_layer_indices:
      raise ValueError("hook_layer_indices must not be empty.")

    self.encoder: nn.Module = encoder
    self.decoder: nn.Module = decoder
    self.patch_size: int = encoder.config.patch_size
    self._features: dict[str, torch.Tensor] = {}
    self.hook_layer_indices: tuple[int, ...] = hook_layer_indices
    self.feat_names: list[str] = [f"feat{i}" for i in self.hook_layer_indices]
    self._hook_handles: list[Any] = []

    def get_activation(name: str) -> Callable[[nn.Module, Any, Any], None]:
      """Returns a hook function to capture activations."""
      def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        del module, inputs
        if isinstance(output, tuple):
          output = output[0]
        self._features[name] = output
      return hook

    for i, name in zip(self.hook_layer_indices, self.feat_names):
      handle = self.encoder.encoder.layer[i].register_forward_hook(
          get_activation(name)
      )
      self._hook_handles.append(handle)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the segmentation model.

    Args:
        x: Input image tensor.

    Returns:
        Output segmentation map from the decoder.

    Raises:
        RuntimeError: If a feature expected from a hook is not found,
          indicating an issue with the hook setup or encoder structure.
    """
    self._features = {}

    _ = self.encoder(x)

    features = []

    b = x.shape[0]
    h = x.shape[2] // self.patch_size
    w = x.shape[3] // self.patch_size

    for name in self.feat_names:
      if name not in self._features:
        raise RuntimeError(
            f"Missing hooked feature '{name}'. Check hook_layer_indices and "
            "encoder structure."
        )

      feat = self._features[name]

      if feat.shape[1] == h * w + 1:
        feat = feat[:, 1:, :]

      if feat.shape[1] != h * w:
        raise ValueError(
            f"Feature '{name}' has incompatible token count {feat.shape[1]} "
            f"for spatial size ({h}, {w})."
        )

      feat = feat.transpose(1, 2).reshape(b, -1, h, w)
      features.append(feat)

    return self.decoder(features)




