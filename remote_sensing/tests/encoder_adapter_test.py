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

from typing import Iterator
import unittest

from remote_sensing.models import encoder_adapter
import torch
from torch import nn


class MockEmbeddings(nn.Module):
  """A mock embedding layer for testing purposes."""

  def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
    super().__init__()
    self.patch_embeddings = nn.Module()
    self.patch_embeddings.projection = nn.Conv2d(
        in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
    )
    self.patch_embeddings.num_channels = in_channels


class MockEncoderLayer(nn.Module):
  """A mock encoder layer for testing purposes.

  This layer contains a simple ModuleList of two linear layers.
  """

  def __init__(self, hidden_size: int):
    super().__init__()
    self.linears = nn.ModuleList(
        [nn.Linear(hidden_size, hidden_size) for _ in range(2)]
    )

  def parameters(self) -> Iterator[nn.Parameter]:
    params = []
    for linear in self.linears:
      params.extend(linear.parameters())
    return iter(params)


class MockEncoderModule(nn.Module):
  """A mock encoder module for testing purposes.

  This module contains mock embeddings and a series of mock encoder layers.
  """

  def __init__(
      self,
      num_channels: int = 3,
      hidden_size: int = 8,
      patch_size: int = 2,
      num_layers: int = 2,
  ):
    super().__init__()
    self.config = lambda: None
    self.config.num_channels = num_channels
    self.config.hidden_size = hidden_size

    self.embeddings = MockEmbeddings(num_channels, hidden_size, patch_size)
    self.encoder = nn.Module()
    self.encoder.layer = nn.ModuleList(
        [MockEncoderLayer(hidden_size) for _ in range(num_layers)]
    )

  def parameters(self) -> Iterator[nn.Parameter]:
    params = list(self.embeddings.parameters())
    for layer in self.encoder.layer:
      params.extend(layer.parameters())
    return iter(params)


class TestEncoderAdapter(unittest.TestCase):
  """Tests for the encoder adapter free functions."""

  def test_patch_multispectral_no_change(self):
    """Tests `patch_encoder_to_multispectral` when target and source channels are the same."""
    encoder = MockEncoderModule(num_channels=3)
    old_proj = encoder.embeddings.patch_embeddings.projection

    encoder_adapter.patch_encoder_to_multispectral(
        encoder, target_channels="RGB", source_channels="RGB"
    )

    self.assertEqual(encoder.config.num_channels, 3)

    # Invariant: Output projection is unchanged
    image = torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(
        old_proj(image), encoder.embeddings.patch_embeddings.projection(image)
    )

  def test_adapt_multispectral_add_channel(self):
    """Tests adapt_multispectral when adding a new channel.

    The new channel's weights should be the mean of the existing channels.
    """
    encoder = MockEncoderModule(num_channels=3)
    old_proj = encoder.embeddings.patch_embeddings.projection

    encoder_adapter.patch_encoder_to_multispectral(
        encoder, target_channels="RGBN", source_channels="RGB"
    )
    new_proj = encoder.embeddings.patch_embeddings.projection

    self.assertEqual(encoder.config.num_channels, 4)
    # Invariant 1: Appending a zero channel does not change the original output.
    rgb_image = torch.randn(2, 3, 4, 4)
    rgbn_zeros = torch.cat([rgb_image, torch.zeros(2, 1, 4, 4)], dim=1)
    torch.testing.assert_close(old_proj(rgb_image), new_proj(rgbn_zeros))

    # Invariant 2: Because the new channel's weights are the mean of the
    # original weights, passing a value N into the new channel is equivalent to
    # passing N/3 into all 3 original channels.
    n_channel = torch.randn(2, 1, 4, 4)
    rgbn_n_only = torch.cat([torch.zeros(2, 3, 4, 4), n_channel], dim=1)
    rgb_n_distributed = n_channel.expand(-1, 3, -1, -1) / 3
    torch.testing.assert_close(
        new_proj(rgbn_n_only), old_proj(rgb_n_distributed)
    )

  def test_adapt_multispectral_reorder_channels(self):
    """Tests adapt_multispectral when reordering existing channels."""
    encoder = MockEncoderModule(num_channels=3)
    old_proj = encoder.embeddings.patch_embeddings.projection

    encoder_adapter.patch_encoder_to_multispectral(
        encoder, target_channels="BGR", source_channels="RGB"
    )
    new_proj = encoder.embeddings.patch_embeddings.projection

    self.assertEqual(encoder.config.num_channels, 3)

    # Invariant: A BGR image passed to the BGR encoder yields the same output
    # as the original RGB image passed to the original RGB encoder.
    rgb_image = torch.randn(2, 3, 4, 4)
    bgr_image = rgb_image[:, [2, 1, 0], :, :]

    torch.testing.assert_close(old_proj(rgb_image), new_proj(bgr_image))

  def test_adapt_multispectral_channel_mismatch_error(self):
    """Tests that adapt_multispectral raises an error on channel mismatch."""
    encoder = MockEncoderModule(num_channels=4)
    with self.assertRaises(ValueError):
      encoder_adapter.patch_encoder_to_multispectral(
          encoder, target_channels="RGBN", source_channels="RGB"
      )

  def test_freeze_encoder_all(self):
    """Tests freezing all layers and embeddings in the encoder."""
    encoder = MockEncoderModule(num_layers=4)
    encoder_adapter.freeze_encoder(
        encoder, num_unfrozen_layers=0, freeze_patch_projection=True
    )

    for param in encoder.parameters():
      self.assertFalse(param.requires_grad)

  def test_freeze_encoder_unfreeze_layers(self):
    """Tests freezing the encoder while keeping some trailing layers unfrozen."""
    encoder: MockEncoderModule = MockEncoderModule(num_layers=12)
    num_unfrozen: int = 3
    # Freeze the encoder, keeping the last 'num_unfrozen' layers trainable.
    # The patch embedding layer is also frozen.
    encoder_adapter.freeze_encoder(
        encoder, num_unfrozen_layers=num_unfrozen, freeze_patch_projection=True
    )

    # Embeddings should be frozen
    for param in encoder.embeddings.parameters():
      self.assertFalse(param.requires_grad)

    # First N - num_unfrozen layers should be frozen
    for i in range(12 - num_unfrozen):
      for param in encoder.encoder.layer[i].parameters():
        self.assertFalse(param.requires_grad)

    # Last num_unfrozen layers should NOT be frozen
    for i in range(12 - num_unfrozen, 12):
      for param in encoder.encoder.layer[i].parameters():
        self.assertTrue(param.requires_grad)

  def test_freeze_encoder_unfreeze_embedding(self):
    """Tests freezing the encoder layers but not the embedding layer."""
    encoder = MockEncoderModule(num_layers=4)
    encoder_adapter.freeze_encoder(
        encoder, num_unfrozen_layers=0, freeze_patch_projection=False
    )

    # Embedding projection should NOT be frozen
    for param in encoder.embeddings.patch_embeddings.projection.parameters():
      self.assertTrue(param.requires_grad)

    # Other layers should be frozen
    for i in range(4):
      for param in encoder.encoder.layer[i].parameters():
        self.assertFalse(param.requires_grad)

  def test_freeze_encoder_invalid_unfrozen_layers(self):
    """Tests that freeze_encoder raises an error for invalid num_unfrozen_layers."""
    encoder = MockEncoderModule(num_layers=4)
    with self.assertRaises(ValueError):
      encoder_adapter.freeze_encoder(encoder, num_unfrozen_layers=-1)


if __name__ == "__main__":
  unittest.main()
