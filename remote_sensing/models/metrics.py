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

"""Utilities for computing metrics over datasets."""

import abc
import collections
import re
from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
import torch
import typing_extensions

# A segmentation loss function, taking logits, targets, and optional weight,
# and returning a scalar loss value.
SegmentationLossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
]


class _AccumulatedMetrics(abc.ABC):
  """Base class for accumulating metrics over batches.

  Inherited classes should implement 3 main functions:
  - __init__, in which they initialize their metrics
  - add_batch, which accumulates metrics from another batch
  - compute_epoch_metrics, which summarizes all the accumulated metrics and
    outputs the per-epoch metrics.
  """

  @abc.abstractmethod
  def add_batch(self, *args, **kwargs):
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_epoch_metrics(self) -> dict[str, Any]:
    raise NotImplementedError()


class _SegmentationMetrics(_AccumulatedMetrics):
  """Accumulates metrics for segmentation tasks.

  Usage:
    metrics = _SegmentationMetrics(
        {0: "bg", 1: "road"},
        {"Xent": losses.SegmentationCrossEntropyLoss(),
         "Dice": losses.SegmentationDiceLoss()},
    )
    for batch in dataloader:
      logits = model(batch["image"])
      metrics.add_batch(logits, batch['label'], batch['weight'])
    print(metrics.compute_epoch_metrics())
  """

  def __init__(
      self,
      class_names: dict[int, str],
      losses: dict[str, SegmentationLossFn],
  ):
    self.class_names = class_names
    self.num_classes = len(class_names)
    self.losses = losses

    # Accumulated metrics:

    # Total number of batches.
    self.batches_count = 0

    # The sum of the loss values for each loss function.
    self.loss_values = collections.defaultdict(float)

    # The sum of the soft (probabilistic) confusion matrix.
    self.soft_confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    # The sum of the hard (discrete) confusion matrix.
    self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

  @typing_extensions.override
  @torch.inference_mode()
  def add_batch(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ):
    """Adds a batch of data to the accumulated metrics.

    Args:
      logits: The logits from the model, of shape ([B...], C, H, W).
      targets: The targets probabilities, of shape ([B...], C, H, W).
      weight: The weights, of shape ([B...], 1, H, W), or None.
    """
    for loss_name, loss_fn in self.losses.items():
      self.loss_values[loss_name] += loss_fn(logits, targets, weight).item()
    self.batches_count += 1
    probs = torch.softmax(logits, dim=-3).float()

    if weight is None:
      weight = torch.ones(
          (*probs.shape[:-3], 1, *probs.shape[-2:]), device=probs.device
      )

    self.soft_confusion_matrix += (
        torch.einsum(
            "...chw,...dhw->...cd", targets.float() * weight.float(), probs
        )
        .sum(dim=tuple(range(logits.dim() - 3)))
        .numpy(force=True)
    )

    pred = torch.argmax(logits, dim=-3).numpy(force=True)
    label = torch.argmax(targets, dim=-3).numpy(force=True)
    mask = (weight > 0).squeeze(-3).numpy(force=True)

    self.confusion_matrix += sklearn_metrics.confusion_matrix(
        label[mask],
        pred[mask],
        labels=np.arange(self.num_classes),
    )

  @typing_extensions.override
  def compute_epoch_metrics(self) -> dict[str, Any]:
    """Computes the metrics for the epoch."""
    per_class_intersection = np.diag(self.soft_confusion_matrix)
    per_class_sum = np.sum(self.soft_confusion_matrix, axis=1) + np.sum(
        self.soft_confusion_matrix, axis=0
    )
    per_class_union = per_class_sum - per_class_intersection
    per_class_tp = np.diag(self.confusion_matrix)
    per_class_fp = np.sum(self.confusion_matrix, axis=0) - per_class_tp
    per_class_fn = np.sum(self.confusion_matrix, axis=1) - per_class_tp
    with np.errstate(divide="ignore", invalid="ignore"):
      dice = 2 * per_class_intersection / per_class_sum
      jaccard = per_class_intersection / per_class_union
      per_class_precision = per_class_tp / (per_class_tp + per_class_fp)
      per_class_recall = per_class_tp / (per_class_tp + per_class_fn)
      f1 = 2 * per_class_tp / (2 * per_class_tp + per_class_fp + per_class_fn)
      iou = per_class_tp / (per_class_tp + per_class_fp + per_class_fn)
      accuracy = np.sum(per_class_tp) / np.sum(self.confusion_matrix)

    res = {}
    for loss_name, loss_value in self.loss_values.items():
      res[f"loss_{loss_name}"] = loss_value / self.batches_count
    res["mean_dice"] = np.nanmean(dice)
    res["mean_jaccard"] = np.nanmean(jaccard)
    res["mean_f1"] = np.nanmean(f1)
    res["mean_iou"] = np.nanmean(iou)
    res["mean_accuracy"] = accuracy
    for c in range(self.num_classes):
      cn = self.class_names[c]
      res[f"class_{cn}_dice"] = dice[c]
      res[f"class_{cn}_jaccard"] = jaccard[c]
      res[f"class_{cn}_iou"] = iou[c]
      res[f"class_{cn}_f1"] = f1[c]
      res[f"class_{cn}_recall"] = per_class_recall[c]
      res[f"class_{cn}_precision"] = per_class_precision[c]
    res["confusion_matrix"] = self.confusion_matrix
    res["soft_confusion_matrix"] = self.soft_confusion_matrix
    return res


class MetricsTracker:
  r"""Tracks metrics over multiple epochs.

  Usage:
    tracker = segmentation_metrics_tracker(
        {0: "bg", 1: "road"},
        {
            "Xent": losses.SegmentationCrossEntropyLoss(),
            "Dice": losses.SegmentationDiceLoss()
        }
    )
    for epoch in range(num_epochs):
      for batch in train_dataloader:
        logits = model(batch["image"])
        tracker.add_batch('Train', logits, batch['label'], batch['weight'])
      for batch in val_dataloader:
        logits = model(batch["image"])
        tracker.add_batch('Val', logits, batch['label'], batch['weight'])
      tracker.next_epoch()
    tracker.filter_and_rename("(.*)_mean_(iou|f1)", "\\0 - \\1").plot()
  """

  def __init__(self, factory: Callable[[], _AccumulatedMetrics]):
    """Initializes the tracker, given a factory for `_AccumulatedMetrics` objects."""
    self.cur_epoch_metrics = collections.defaultdict(factory)
    self.per_epoch_metrics = pd.DataFrame()
    self.epoch = 1

  def add_batch(self, split: str, *args, **kwargs):
    """Adds another batch to the current epoch."""
    self.cur_epoch_metrics[split].add_batch(*args, **kwargs)

  def next_epoch(self):
    """Flushes the current epoch, and reset the metrics accumulation."""
    epoch_metrics = {}
    for split, metrics in self.cur_epoch_metrics.items():
      split_metrics = metrics.compute_epoch_metrics()
      for k, v in split_metrics.items():
        epoch_metrics[f"{split}_{k}"] = v
    new_row_df = pd.DataFrame(index=[self.epoch], data=[epoch_metrics])
    self.per_epoch_metrics = pd.concat([self.per_epoch_metrics, new_row_df])
    self.cur_epoch_metrics.clear()
    self.epoch += 1

  def filter_and_rename(self, regexp_from: str, regexp_to: str) -> pd.DataFrame:
    """Filters only metrics matching `regexp_from` and map them to `regexp_to`."""
    res = pd.DataFrame()
    for col in self.per_epoch_metrics.columns:
      groups = re.fullmatch(regexp_from, col)
      if groups is None:
        continue
      g = groups.groups()
      label = re.sub(r"\\(\d+)", lambda m, g=g: g[int(m.group(1))], regexp_to)
      res[label] = self.per_epoch_metrics[col]
    return res

  def last(self) -> pd.Series:
    """Returns the last epoch metrics."""
    return self.per_epoch_metrics.iloc[-1]


def segmentation_metrics_tracker(
    class_names: dict[int, str],
    losses: dict[str, SegmentationLossFn],
) -> MetricsTracker:
  """Returns a MetricsTracker for segmentation tasks."""
  return MetricsTracker(lambda: _SegmentationMetrics(class_names, losses))
