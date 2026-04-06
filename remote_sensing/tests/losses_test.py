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
from remote_sensing.models import losses
import torch


class LossesTest(absltest.TestCase):

  def test_focal_loss(self):
    logits = torch.randn(2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64)).float()
    loss = losses.sigmoid_focal_loss(
        inputs=logits, targets=targets, alpha=0.25, gamma=2.0, reduction='mean'
    )
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), 0.0)

  def test_dice_loss(self):
    loss_fn = losses.BinarySegmentationDiceLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64)).float()
    loss = loss_fn(logits, targets)
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), 0.0)
    self.assertLessEqual(loss.item(), 1.0)

  def test_combo_loss(self):
    dice_loss = losses.BinarySegmentationDiceLoss()
    loss_fn = losses.CombinedLoss(losses=[dice_loss], weights=[1.0])
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64)).float()
    loss = loss_fn(logits, targets)
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), 0.0)

  def test_loss_backward(self):
    dice_loss = losses.BinarySegmentationDiceLoss()
    loss_fn = losses.CombinedLoss(losses=[dice_loss], weights=[1.0])
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 64, 64)).float()
    loss = loss_fn(logits, targets)
    loss.backward()
    self.assertIsNotNone(logits.grad)


if __name__ == '__main__':
  absltest.main()
