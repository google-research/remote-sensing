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

  def test_segmentation_dice_loss(self):
    # Binary case
    loss_fn_binary = losses.SegmentationDiceLoss()
    logits_binary = torch.randn(2, 1, 64, 64)
    targets_binary = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_binary = loss_fn_binary(logits_binary, targets_binary)
    self.assertIsInstance(loss_binary.item(), float)
    self.assertGreaterEqual(loss_binary.item(), 0.0)
    self.assertLessEqual(loss_binary.item(), 1.0)

    # Multi-class case
    loss_fn_multiclass = losses.SegmentationDiceLoss()
    logits_multiclass = torch.randn(2, 3, 64, 64)
    targets_multiclass = torch.nn.functional.one_hot(
        torch.randint(0, 3, (2, 64, 64)).long(), num_classes=3
    ).permute(0, 3, 1, 2).float()
    loss_multiclass = loss_fn_multiclass(logits_multiclass, targets_multiclass)
    self.assertIsInstance(loss_multiclass.item(), float)
    self.assertGreaterEqual(loss_multiclass.item(), 0.0)
    self.assertLessEqual(loss_multiclass.item(), 1.0)

  def test_segmentation_jaccard_loss(self):
    # Binary case
    loss_fn_binary = losses.SegmentationJaccardLoss()
    logits_binary = torch.randn(2, 1, 64, 64)
    targets_binary = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_binary = loss_fn_binary(logits_binary, targets_binary)
    self.assertIsInstance(loss_binary.item(), float)
    self.assertGreaterEqual(loss_binary.item(), 0.0)
    self.assertLessEqual(loss_binary.item(), 1.0)

    # Multi-class case
    loss_fn_multiclass = losses.SegmentationJaccardLoss()
    logits_multiclass = torch.randn(2, 3, 64, 64)
    targets_multiclass = (
        torch.nn.functional.one_hot(
            torch.randint(0, 3, (2, 64, 64)).long(), num_classes=3
        )
        .permute(0, 3, 1, 2)
        .float()
    )
    loss_multiclass = loss_fn_multiclass(logits_multiclass, targets_multiclass)
    self.assertIsInstance(loss_multiclass.item(), float)
    self.assertGreaterEqual(loss_multiclass.item(), 0.0)
    self.assertLessEqual(loss_multiclass.item(), 1.0)

  def test_segmentation_focal_loss(self):
    # Binary case
    loss_fn_binary = losses.SegmentationFocalLoss()
    logits_binary = torch.randn(2, 1, 64, 64)
    targets_binary = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_binary = loss_fn_binary(logits_binary, targets_binary)
    self.assertIsInstance(loss_binary.item(), float)
    self.assertGreaterEqual(loss_binary.item(), 0.0)

    # Multi-class case
    loss_fn_multiclass = losses.SegmentationFocalLoss()
    logits_multiclass = torch.randn(2, 3, 64, 64)
    targets_multiclass = torch.nn.functional.one_hot(
        torch.randint(0, 3, (2, 64, 64)).long(), num_classes=3
    ).permute(0, 3, 1, 2).float()
    loss_multiclass = loss_fn_multiclass(logits_multiclass, targets_multiclass)
    self.assertIsInstance(loss_multiclass.item(), float)
    self.assertGreaterEqual(loss_multiclass.item(), 0.0)

    weight = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_weighted = loss_fn_multiclass(
        logits_multiclass, targets_multiclass, weight
    )
    self.assertIsInstance(loss_weighted.item(), float)
    self.assertGreaterEqual(loss_weighted.item(), 0.0)

  def test_combo_loss(self):
    dice_loss = losses.SegmentationDiceLoss()
    focal_loss = losses.SegmentationFocalLoss()
    loss_fn = losses.CombinedLoss(
        losses=[dice_loss, focal_loss], weights=[0.5, 0.5]
    )
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), 0.0)

    weight = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_weighted = loss_fn(logits, targets, weight)
    self.assertIsInstance(loss_weighted.item(), float)
    self.assertGreaterEqual(loss_weighted.item(), 0.0)

  def test_loss_backward(self):
    dice_loss = losses.SegmentationDiceLoss()
    focal_loss = losses.SegmentationFocalLoss()
    loss_fn = losses.CombinedLoss(
        losses=[dice_loss, focal_loss], weights=[0.5, 0.5]
    )
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = loss_fn(logits, targets)
    loss.backward()
    self.assertIsNotNone(logits.grad)

  def test_segmentation_lovasz_softmax_loss(self):
    loss_fn = losses.SegmentationLovaszSoftmaxLoss()
    logits = torch.randn(2, 3, 10, 10)
    targets_indices = torch.randint(0, 3, (2, 10, 10)).long()
    targets = torch.nn.functional.one_hot(
        targets_indices, num_classes=3
    ).permute(0, 3, 1, 2).float()
    loss = loss_fn(logits, targets)
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), -1.0)
    self.assertLessEqual(loss.item(), 2.0)

    weight = torch.randint(0, 2, (2, 1, 10, 10)).float()
    loss_weighted = loss_fn(logits, targets, weight)
    self.assertIsInstance(loss_weighted.item(), float)
    self.assertGreaterEqual(loss_weighted.item(), -1.0)
    self.assertLessEqual(loss_weighted.item(), 2.0)

    # Test with perfect prediction
    logits_perfect = targets * 100.0 - (1 - targets) * 100.0
    loss_perfect = loss_fn(logits_perfect, targets)
    self.assertAlmostEqual(loss_perfect.item(), 0.0, places=3)

  def test_segmentation_cross_entropy_loss(self):
    loss_fn = losses.SegmentationCrossEntropyLoss()
    logits = torch.randn(2, 3, 10, 10)
    targets = torch.randint(0, 3, (2, 10, 10)).long()
    weight = torch.randint(0, 2, (2, 1, 10, 10)).float()
    targets_onehot = (
        torch.nn.functional.one_hot(targets, num_classes=3)
        .permute(0, 3, 1, 2)
        .float()
    )
    loss = loss_fn(logits, targets_onehot, weight)
    self.assertIsInstance(loss.item(), float)
    self.assertGreaterEqual(loss.item(), 0.0)

  def test_progressive_combined_loss(self):
    class DummyLoss(torch.nn.Module):

      def __init__(self, value: float):
        super().__init__()
        self.value = value

      def forward(self, logits, targets, weight=None):
        return self.value

    def weights_fn(cur_step: int, total_steps: int) -> list[float]:
      return [1 - cur_step / total_steps, cur_step / total_steps]

    combined_loss = losses.ProgressiveCombinedLoss(
        losses=[DummyLoss(100), DummyLoss(200)],
        weights_provider=weights_fn,
        total_steps=50
    )
    x = torch.ones((1,))  # Dummy input
    for i in range(51):
      self.assertAlmostEqual(combined_loss(x, x).item(), 100 + i * 2, places=3)
      combined_loss.next_step()


if __name__ == '__main__':
  absltest.main()
