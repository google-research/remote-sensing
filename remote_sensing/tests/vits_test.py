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

"""Tests for the remote sensing vits module."""

from absl.testing import absltest
import numpy as np
from remote_sensing.models import vits
import torch
import transformers


class TorchModelsTest(absltest.TestCase):

  def test_vit_smoke_test(self):
    config = transformers.ViTConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=1,
        intermediate_size=32,
        image_size=35,
        patch_size=5,
        num_channels=3,
    )
    model = vits.PretrainedRemoteSensingVit(config)
    for param in model.parameters():
      if param.ndim <= 1:
        torch.nn.init.zeros_(param)
      elif param.ndim == 2:
        torch.nn.init.eye_(param)
      else:
        torch.nn.init.dirac_(param)
    model.eval()
    with torch.no_grad():
      out = model(torch.ones((11, 3, 35, 35), dtype=torch.float32))
    self.assertEqual(out.last_hidden_state.shape, (11, 7 * 7, 16))
    pooled = np.mean(np.array(out.last_hidden_state[0, ...]), axis=0)
    np.testing.assert_allclose(
        pooled,
        # pyformat: disable
        np.array([1.09, 0.90, 0.99, 0.00, -0.06, 0.98, 1.00, 1.00,
                  0.09, -0.1, 0.00, 0.00, -0.06, 0.98, 1.00, 1]),
        # pyformat: enable
        rtol=0.0,
        atol=1e-2,
    )


if __name__ == "__main__":
  absltest.main()
