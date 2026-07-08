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
import numpy as np
from remote_sensing.models import losses
from remote_sensing.models import metrics
import torch


def one_hot_full(val: int) -> torch.Tensor:
  return (
      torch.nn.functional.one_hot(
          torch.full((2, 10, 10), val).long(), num_classes=3
      )
      .permute(0, 3, 1, 2)
      .float()
  )


class MetricsTest(absltest.TestCase):

  def test_segmentation_metrics_tracker(self):
    class_names = {0: "bg", 1: "road", 2: "water"}
    loss_fns = {
        "dice": losses.SegmentationDiceLoss(),
        "xent": losses.SegmentationCrossEntropyLoss(),
    }
    tracker = metrics.segmentation_metrics_tracker(class_names, loss_fns)
    weight = torch.ones((2, 1, 10, 10)).float()
    tracker.add_batch("train", one_hot_full(0), one_hot_full(0), weight)
    tracker.add_batch("val", one_hot_full(0), one_hot_full(0), weight)
    tracker.add_batch("val", one_hot_full(0), one_hot_full(1), weight)
    tracker.add_batch("val", one_hot_full(2), one_hot_full(1), weight)
    tracker.next_epoch()
    m = tracker.per_epoch_metrics

    e = np.e
    test_metric = lambda k, v: np.testing.assert_array_almost_equal(m[k], [v])
    test_metric("train_class_bg_iou", 1)
    test_metric("train_class_road_iou", np.nan)
    test_metric("train_class_water_iou", np.nan)
    test_metric("train_class_bg_f1", 1)
    test_metric("train_class_road_f1", np.nan)
    test_metric("train_class_water_f1", np.nan)
    test_metric("train_class_bg_dice", e / (e + 1))
    test_metric("train_class_road_dice", 0)
    test_metric("train_class_water_dice", 0)
    test_metric("train_loss_dice", 1 - e / 3 / (e + 1))
    test_metric("train_loss_xent", 2 * np.log(e + 2) - 2)
    test_metric("train_mean_dice", e / 3 / (e + 1))
    test_metric("train_mean_jaccard", e / 3 / (e + 2))
    test_metric("train_mean_iou", 1)
    test_metric("train_mean_f1", 1)
    test_metric("train_mean_accuracy", 1)
    test_metric("val_class_bg_iou", 0.5)
    test_metric("val_class_road_iou", 0.0)
    test_metric("val_class_water_iou", 0.0)
    test_metric("val_class_bg_precision", 0.5)
    test_metric("val_class_road_precision", np.nan)
    test_metric("val_class_water_precision", 0.0)
    test_metric("val_class_bg_recall", 1.0)
    test_metric("val_class_road_recall", 0.0)
    test_metric("val_class_water_recall", np.nan)
    test_metric("val_mean_accuracy", 1 / 3.0)

  def test_multiple_epochs(self):
    class_names = {0: "bg", 1: "road", 2: "water"}
    tracker = metrics.segmentation_metrics_tracker(class_names, {})
    weight = torch.ones((2, 1, 10, 10)).float()
    tracker.add_batch("train", one_hot_full(0), one_hot_full(0), weight)
    tracker.next_epoch()
    tracker.add_batch("train", one_hot_full(1), one_hot_full(0), weight)
    tracker.next_epoch()
    m = tracker.filter_and_rename("train_class_(.*)_iou", "\\0 IOU")
    test_metric = lambda k, v: np.testing.assert_array_almost_equal(m[k], v)
    test_metric("bg IOU", [1, 0])
    test_metric("road IOU", [np.nan, 0])
    test_metric("water IOU", [np.nan, np.nan])
    self.assertEqual(tracker.last()["train_confusion_matrix"].shape, (3, 3))


if __name__ == "__main__":
  absltest.main()
