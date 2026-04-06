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

"""Utilities for handling and adapting pretrained encoders.

This module provides helper functions to adapt pretrained PyTorch encoders for
multispectral inputs and to freeze layers for fine-tuning.
"""

import torch
from torch import nn
import transformers


def patch_encoder_to_multispectral(
    encoder: transformers.ViTModel,
    *,
    target_channels: str = "RGBN",
    source_channels: str = "RGB",
) -> None:
  """Patches the encoder configuration and weights to support multispectral inputs.

  This function modifies the encoder that was configured and trained for RGB
  inputs to work with multispectral inputs by reconfiguring the patch projection
  layer and modifying the projection layer weights.

  Args:
      encoder: The pretrained PyTorch encoder.
      target_channels: String representing the desired channel order, e.g.,
        "RGBN".
      source_channels: String representing the original channel order, e.g.,
        "RGB".

  Raises:
      ValueError: If the encoder's channel count doesn't match
      source_channels.
  """
  if target_channels == source_channels:
    return

  if encoder.config.num_channels != len(source_channels):
    raise ValueError(
        f"Encoder has {encoder.config.num_channels} channels, but"
        f" source_channels='{source_channels}' has length"
        f" {len(source_channels)}."
    )

  old_proj = encoder.embeddings.patch_embeddings.projection
  new_proj = nn.Conv2d(
      in_channels=len(target_channels),
      out_channels=old_proj.out_channels,
      kernel_size=old_proj.kernel_size,
      stride=old_proj.stride,
      padding=old_proj.padding,
      bias=(old_proj.bias is not None),
  )

  with torch.no_grad():
    mean_weights = torch.mean(old_proj.weight, dim=1)
    for new_idx, channel in enumerate(target_channels):
      if channel in source_channels:
        old_idx = source_channels.index(channel)
        new_proj.weight[:, new_idx] = old_proj.weight[:, old_idx]
      else:
        new_proj.weight[:, new_idx] = mean_weights
    if old_proj.bias is not None:
      new_proj.bias.copy_(old_proj.bias)

  encoder.embeddings.patch_embeddings.projection = new_proj
  encoder.config.num_channels = len(target_channels)
  encoder.embeddings.patch_embeddings.num_channels = len(target_channels)


def freeze_encoder(
    encoder: transformers.ViTModel,
    num_unfrozen_layers: int,
    freeze_patch_projection: bool = False,
) -> None:
  """Freezes encoder layers except for trailing transformer blocks.

  Args:
      encoder: The pretrained PyTorch encoder.
      num_unfrozen_layers: Number of trailing transformer blocks to leave
        unfrozen.
      freeze_patch_projection: If False, leaves patch embedding projection
        unfrozen.
  """
  if num_unfrozen_layers < 0:
    raise ValueError("num_unfrozen_layers must be non-negative.")

  for p in encoder.parameters():
    p.requires_grad = False

  if num_unfrozen_layers > 0:
    for block in encoder.encoder.layer[-num_unfrozen_layers:]:
      for p in block.parameters():
        p.requires_grad = True

  if not freeze_patch_projection:
    for p in encoder.embeddings.patch_embeddings.projection.parameters():
      p.requires_grad = True
