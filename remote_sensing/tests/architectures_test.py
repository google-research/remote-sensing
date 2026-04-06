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


from absl.testing import absltest
from remote_sensing.models import architectures
import torch
from torch import nn


class FakeConfig:
  def __init__(self, hidden_size=64, patch_size=16, num_channels=3):
    self.hidden_size = hidden_size
    self.patch_size = patch_size
    self.num_channels = num_channels


class FakeEncoderLayer(nn.Module):
  def forward(self, x):
    return x


class FakeEncoderLayers(nn.Module):
  def __init__(self, num_layers=4):
    super().__init__()
    self.layer = nn.ModuleList([FakeEncoderLayer() for _ in range(num_layers)])

  def forward(self, x):
    for l in self.layer:
      x = l(x)
    return x


class FakePretrainedVit(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.encoder = FakeEncoderLayers()

  def forward(self, x):
    # Fake embedding: x is (B, C, H, W) -> we need (B, sequence_length, dim)
    b = x.shape[0]
    h = x.shape[2] // self.config.patch_size
    w = x.shape[3] // self.config.patch_size
    # Fake sequence: +1 for CLS token
    fake_tokens = torch.randn(b, h*w + 1, self.config.hidden_size)
    return self.encoder(fake_tokens)

  @classmethod
  def from_pretrained(cls):
    config = FakeConfig()
    return cls(config)


class ArchitecturesTest(absltest.TestCase):

  def test_conv_block(self):
    block = architectures.ConvBlock(in_c=3, out_c=8, groups=4)
    x = torch.randn(2, 3, 32, 32)
    out = block(x)
    self.assertEqual(out.shape, (2, 8, 32, 32))

  def test_vit_unet_decoder(self):
    decoder = architectures.ViTUNetDecoder(
        encoder_dim=64,
        decoder_channels=(128, 64, 32, 16, 8),
        output_dims=2,
    )
    b, c, h, w = 2, 64, 8, 8
    x1 = torch.randn(b, c, h, w)
    x2 = torch.randn(b, c, h, w)
    x3 = torch.randn(b, c, h, w)
    x4 = torch.randn(b, c, h, w)
    x5 = torch.randn(b, c, h, w)

    features = [x1, x2, x3, x4, x5]
    out = decoder(features)

    # 4 upsampling steps with scale factor 2: h*2*2*2*2 = h*16
    self.assertEqual(out.shape, (2, 2, 128, 128))


if __name__ == '__main__':
  absltest.main()
