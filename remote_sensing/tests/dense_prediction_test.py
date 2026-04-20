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
from absl.testing import parameterized
import numpy as np
from remote_sensing.models import dense_prediction
import torch
import transformers


class DensePredictionTest(parameterized.TestCase):

  @parameterized.parameters(
      {"pos_emb_type": "sincos2d"},
      {"pos_emb_type": "trainable"},
  )
  def test_decoder_smoke_test(self, pos_emb_type):
    torch.manual_seed(1)
    config = dense_prediction.VitDecoderConfig(
        output_dims=8,
        encoder_hidden_size=16,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=32,
        patch_size=4,
        num_channels=3,
        pos_emb_type=pos_emb_type,
    )
    decoder = dense_prediction.ViTDecoder(config)
    decoder.eval()
    with torch.no_grad():
      hidden_states = torch.ones((5, 64, 16), dtype=torch.float32)
      out = decoder(hidden_states)
    self.assertEqual(out.shape, (5, 32, 32, 8))
    # Check that outputs are bounded by Tanh [-1, 1]
    self.assertTrue(torch.all(out >= -1.0))
    self.assertTrue(torch.all(out <= 1.0))
    # Check for finite values
    self.assertTrue(torch.all(torch.isfinite(out)))

  def test_init_invalid_image_size(self):
    with self.assertRaisesRegex(
        ValueError, "Image size .* must be divisible by patch size .*"
    ):
      config = dense_prediction.VitDecoderConfig(
          output_dims=8,
          encoder_hidden_size=16,
          hidden_size=12,
          num_hidden_layers=1,
          num_attention_heads=1,
          intermediate_size=32,
          image_size=30,  # Not divisible by 4
          patch_size=4,
          num_channels=3,
      )
      dense_prediction.ViTDecoder(config)

  def test_forward_invalid_num_patches(self):
    config = dense_prediction.VitDecoderConfig(
        output_dims=8,
        encoder_hidden_size=16,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=32,
        patch_size=4,
        num_channels=3,
    )
    decoder = dense_prediction.ViTDecoder(config)
    decoder.eval()
    with torch.no_grad():
      # Expected 64 patches (32/4)^2. Providing 63.
      hidden_states = torch.ones((1, 63, 16), dtype=torch.float32)
      with self.assertRaisesRegex(ValueError, "63.*must be a perfect square"):
        decoder(hidden_states)

  def test_forward_different_num_patches(self):
    config = dense_prediction.VitDecoderConfig(
        output_dims=8,
        encoder_hidden_size=16,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=32,
        patch_size=4,
        num_channels=3,
    )
    decoder = dense_prediction.ViTDecoder(config)
    decoder.eval()
    with torch.no_grad():
      # Expected 64 patches (32/4)^2. Providing 25.
      hidden_states = torch.ones((1, 25, 16), dtype=torch.float32)
      with torch.no_grad():
        out = decoder(hidden_states)
        # Each side has 5 patches and 4 pixels, so 20x20 pixels.
        self.assertEqual(out.shape, (1, 20, 20, 8))

  def test_vit_dense_prediction_model_smoke_test(self):
    torch.manual_seed(1)
    image_size = 32
    encoder_hidden_size = 16
    encoder_config = transformers.ViTConfig(
        hidden_size=encoder_hidden_size,
        num_hidden_layers=2,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=image_size,
        patch_size=4,
        num_channels=3,
        hidden_dropout_prob=0.0,
    )
    decoder_config = dense_prediction.VitDecoderConfig(
        output_dims=8,
        encoder_hidden_size=encoder_hidden_size,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=image_size,
        patch_size=4,
        num_channels=3,
        hidden_dropout_prob=0.0,
        pos_emb_type="sincos2d",
    )
    composite_config = dense_prediction.EncoderDecoderConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    model = dense_prediction.ViTEncoderDecoderModel(composite_config)
    model.eval()
    with torch.no_grad():
      out = model(torch.ones((4, 3, 32, 32), dtype=torch.float32))
    self.assertEqual(out.shape, (4, 32, 32, 8))
    # Check that outputs are bounded by Tanh [-1, 1]
    self.assertTrue(torch.all(out >= -1.0))
    self.assertTrue(torch.all(out <= 1.0))
    # Check for finite values
    self.assertTrue(torch.all(torch.isfinite(out)))

  def test_skip_unet_decoder_config(self):
    config = dense_prediction.SkipUNetDecoderConfig(
        output_dims=10,
        encoder_hidden_size=768,
        decoder_dims=(512, 256, 128),
        skip_connections=(2, 4, 6),
    )
    self.assertEqual(config.model_type, "unet_dense_prediction_decoder")
    self.assertEqual(config.output_dims, 10)
    self.assertEqual(config.encoder_hidden_size, 768)
    self.assertEqual(config.decoder_dims, (512, 256, 128))
    self.assertEqual(config.skip_connections, (2, 4, 6))

  def test_skip_unet_decoder_model_smoke_test(self):
    torch.manual_seed(1)
    image_size = 128
    num_encoder_layers = 5
    encoder_config = transformers.ViTConfig(
        num_hidden_layers=num_encoder_layers,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=image_size,
        hidden_dropout_prob=0.0,
        hidden_size=64,
        patch_size=16,
        num_channels=4,
    )
    decoder_config = dense_prediction.SkipUNetDecoderConfig(
        encoder_hidden_size=64,
        decoder_dims=(128, 64, 32, 16, 8),
        skip_connections=(0, 1, 2, 3, 4),  # Indices within num_encoder_layers
        output_dims=7,
    )
    composite_config = dense_prediction.EncoderDecoderConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    model = dense_prediction.ViTUNetSegmentationModel(composite_config)
    model.eval()
    with torch.no_grad():
      x = torch.randn((2, 4, image_size, image_size), dtype=torch.float32)
      out = model(x)

    self.assertEqual(out.shape, (2, 7, image_size, image_size))
    # Check for finite values
    self.assertTrue(torch.all(torch.isfinite(out)))

  @parameterized.parameters(
      {"shape": (7, 10)},
      {"shape": (70, 100)},
      {"shape": (13, 17)},
  )
  def test_strided_inference(self, shape):
    class DummyModel(torch.nn.Module):

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (1, 3, 7, 10)
        return torch.concat(
            [x + 1, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))], dim=1
        )

    strided = dense_prediction.StridedInference(
        model=DummyModel(),
        model_in_height=7,
        model_in_width=10,
    )
    data_in = np.random.rand(1, 3, shape[0], shape[1])
    with torch.no_grad():
      out = strided(torch.from_numpy(data_in)).numpy()
    np.testing.assert_allclose(out[:, :3, :, :], data_in + 1)
    np.testing.assert_allclose(
        out[:, 3:, :, :], np.ones((1, 1, shape[0], shape[1]))
    )


if __name__ == "__main__":
  absltest.main()
