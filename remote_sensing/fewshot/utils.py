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

"""Utilities for the few-shot retrieval pipeline."""

import dataclasses
import logging
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from sklearn import metrics


class InvalidTrainLabelsError(ValueError):
  """Raised when train labels are invalid (all zeros or all ones)."""


def calculate_labels_balance_score(labels: np.ndarray) -> float:
  """Calculates a balance score for a numpy array of binary labels.

  The score is 1.0 for a perfect balance (equal numbers of 0s and 1s) and
  0.0 for complete imbalance (all 0s or all 1s).

  Args:
      labels: A numpy array containing binary labels (0s and 1s).

  Returns:
      A float representing the balance score, from 0.0 to 1.0.
  """
  # Get the total number of labels.
  n = labels.size

  # If the array is empty, we can define the score as 0.
  if n == 0:
    return 0.0

  # Efficiently count the number of 1s.
  # For a binary array, this is the sum of all elements.
  count_ones = np.sum(labels)

  # The number of 0s is the total size minus the count of 1s.
  count_zeros = n - count_ones

  # Calculate the score using the formula: 1 - |diff| / n
  score = 1.0 - (abs(count_ones - count_zeros) / n)

  return score


def crop_with_margin(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    margin_ratio: float = 0.1,
    mark_bounding_box: bool = False,
    max_crop_area: Optional[int] = None,
) -> np.ndarray:
  """Crops an image based on a bounding box with an added margin.

  The function is intended for visualizing the crops presented to the "user".
  The margin is added to the crop to allow for better context for the user to
  evaluate the crop.

  Args:
      image: The input image.
      bbox: A tuple (y0, x0, y1, x1) representing the bounding box in normalized
        coordinates (0-1).
      margin_ratio: The margin ratio relative to the crop size.
      mark_bounding_box: Whether to mark the bounding box in the crop. Useful
        for debugging and visualization.
      max_crop_area: The maximum crop area to use. If the crop area is larger
        than this, it will be proportionally scaled down to fit within the
        maximum area.

  Returns:
      The cropped image with the added margin.
  """
  width, height = image.shape[:2]
  y0, x0, y1, x1 = bbox

  # Convert normalized coordinates to pixel coordinates
  x0_px = int(x0 * width)
  y0_px = int(y0 * height)
  x1_px = int(x1 * width)
  y1_px = int(y1 * height)

  # Calculate crop size
  crop_width = x1_px - x0_px
  crop_height = y1_px - y0_px

  # Calculate margin in pixels
  margin_x = int(crop_width * margin_ratio)
  margin_y = int(crop_height * margin_ratio)

  # Adjust bounding box with margin
  x0_margin = max(0, x0_px - margin_x)
  y0_margin = max(0, y0_px - margin_y)
  x1_margin = min(width, x1_px + margin_x)
  y1_margin = min(height, y1_px + margin_y)

  # Crop the image
  cropped_image = image[y0_margin:y1_margin, x0_margin:x1_margin].copy()

  if mark_bounding_box:
    # Draw a red bounding box on the cropped image
    cropped_image = cv2.rectangle(
        cropped_image,
        (x0_px - x0_margin, y0_px - y0_margin),
        (x1_px - x0_margin, y1_px - y0_margin),
        color=(255, 0, 0),
        thickness=1,
    )

  if max_crop_area is not None:
    # Scale the image if it is larger than the maximum crop size
    crop_area = cropped_image.shape[0] * cropped_image.shape[1]
    if crop_area > max_crop_area:
      scale_factor = np.sqrt(max_crop_area / crop_area)
      cropped_image = cv2.resize(
          cropped_image,
          (
              int(cropped_image.shape[1] * scale_factor),
              int(cropped_image.shape[0] * scale_factor),
          ),
          interpolation=cv2.INTER_AREA,
      )

  return cropped_image


def intersection_over_union(box1: np.ndarray, box2: np.ndarray) -> float:
  """Calculates the Intersection over Union (IoU) of two bounding boxes.

  Args:
      box1: Bounding box [y1, x1, y2, x2].
      box2: Bounding box [y1, x1, y2, x2].

  Returns:
      The IoU score.
  """
  y1 = max(box1[0], box2[0])
  x1 = max(box1[1], box2[1])
  y2 = min(box1[2], box2[2])
  x2 = min(box1[3], box2[3])

  intersection_width = max(0, x2 - x1)
  intersection_height = max(0, y2 - y1)
  intersection_area = intersection_width * intersection_height

  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

  union_area = box1_area + box2_area - intersection_area

  if union_area == 0:
    return 0.0

  return intersection_area / union_area


@dataclasses.dataclass(frozen=True)
class ModelResults:
  """Contains the aggregated results across the entire dataset."""

  # Original scores for all predictions. Shape: (N,) where N is the number of
  # items.
  scores: np.ndarray

  # Binary predictions after model. Shape: (N,) where N is the number of items.
  predictions: np.ndarray

  # Binary {0, 1} ground truth labels of all items. Shape: (N,) where N is the
  # number of items.
  labels: np.ndarray

  # Total count of unmatched GTs across dataset
  false_negatives_count: int

  # Model embeddings for all items. Shape: (N, D) where N is the number of items
  # and D is the dimension of the embedding.
  embeddings: Optional[np.ndarray] = None

  # Model crops. Useful for visualization and debugging.
  crops: Optional[Sequence[np.ndarray]] = None


class PrecisionAtRecallCalculator:
  """Calculates recall, precision and threshold for a given target precision.

  This class manages the precision-recall curve computation and helps in
  identifying operating points (thresholds) that achieve specific precision
  targets.

  The precision-recall curve is calculated only once upon initialization.
  """

  def __init__(
      self,
      y_true: np.ndarray,
      y_scores: np.ndarray,
      false_negatives_count: int,
  ):
    self._y_true = y_true
    self._y_scores = y_scores
    self._precision, self._recall, self._thresholds = (
        metrics.precision_recall_curve(y_true, y_scores)
    )
    self._false_negatives_count = false_negatives_count

  def get_average_precision(
      self,
  ) -> float:
    """Calculates the average precision based on the precision-recall curve.

    Returns:
      A average precision.
    """

    # Fix the recall to account for false negatives.
    # Note:
    # precision = TP / (TP + FP) and therefore not affected by FN.
    # recall = TP / (TP + FN)
    positives = np.sum(self._y_true)
    fixed_recall = (
        self._recall * positives / (positives + self._false_negatives_count)
    )

    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve.
    # Due to numerical error, we can get `-0.0` and we therefore clip it.
    return float(
        max(
            0.0,
            -np.sum(np.diff(fixed_recall) * np.array(self._precision)[:-1]),
        )
    )

  def get_recall_at_precision(
      self,
      target_precision: float,
  ) -> tuple[float, float, float]:
    """Calculates the recall value for a given target precision.

    Args:
      target_precision: The desired precision level (e.g., 0.8).

    Returns:
      A tuple containing (recall, actual_precision, threshold).
    """
    # Find the indices where precision is greater than or equal to the target
    valid_indices = np.where(self._precision >= target_precision)[0]

    if valid_indices.size < 2:
      # If there are no indices where precision is greater than or equal to the
      # target, raise an error. Note that a single index means that the recall
      # is either 0 or that there are no positives in the dataset therefore not
      # allowed either.
      if not np.any(self._y_true):
        raise ValueError(
            "No positives in dataset. Target precision not calculated."
        )

      raise ValueError(
          f"Target precision {target_precision} achieved at recall 0."
      )

    # The first index in valid_indices corresponds to the point with the highest
    # recall that still meets the precision target.
    first_valid_index = valid_indices[0]

    # Get the corresponding recall, precision, and threshold
    found_recall = self._recall[first_valid_index]
    found_precision = self._precision[first_valid_index]

    # The 'thresholds' array is one element shorter than 'precision' and
    # 'recall'.
    found_threshold = self._thresholds[max(first_valid_index - 1, 0)]

    # Fix the recall to account for false negatives.
    # Note:
    # precision = TP / (TP + FP) and therefore not affected by FN.
    # recall = TP / (TP + FN)
    positives = np.sum(self._y_true)
    fixed_recall = (
        found_recall * positives / (positives + self._false_negatives_count)
    )

    return (
        float(fixed_recall),
        float(found_precision),
        float(found_threshold),
    )


def classification_report(
    few_shot_result: ModelResults,
    zero_shot_result: ModelResults,
) -> dict[str, Any]:
  """Returns a classification report for the given models results."""

  scalars = {}
  for result, model_name in zip(
      [zero_shot_result, few_shot_result], ["ZS", "FS"]
  ):

    # Concatenate the test predictions with the false negatives.
    labels_with_false_negatives = np.concatenate(
        [result.labels, np.ones(result.false_negatives_count, dtype=int)]
    )
    predictions_with_false_negatives = np.concatenate([
        result.predictions,
        np.zeros(result.false_negatives_count, dtype=int),
    ])

    report = metrics.classification_report(
        labels_with_false_negatives,
        predictions_with_false_negatives,
        output_dict=True,
    )

    scalars.update({
        f"{model_name}_false_negatives_count": result.false_negatives_count,
        f"{model_name}_recall_0": report["0"]["recall"],
        f"{model_name}_precision_0": report["0"]["precision"],
        f"{model_name}_support_0": report["0"]["support"],
        f"{model_name}_f1_score_0": report["0"]["f1-score"],
        f"{model_name}_recall": report["1"]["recall"],
        f"{model_name}_precision": report["1"]["precision"],
        f"{model_name}_support": report["1"]["support"],
        f"{model_name}_f1_score": report["1"]["f1-score"],
        f"{model_name}_accuracy": report["accuracy"],
    })

    pr_calculator = PrecisionAtRecallCalculator(
        y_true=result.labels,
        y_scores=result.scores,
        false_negatives_count=result.false_negatives_count,
    )
    for precision_target in (0.5, 0.6, 0.7, 0.8, 0.9):
      try:
        # Add the recall, etc. for the given precision target.
        recall, precision, threshold = pr_calculator.get_recall_at_precision(
            precision_target
        )
        scalars[f"{model_name}_R@P{precision_target}"] = recall
        scalars[f"{model_name}_P@P{precision_target}"] = precision
        scalars[f"{model_name}_T@P{precision_target}"] = threshold
      except ValueError as e:
        logging.warning(
            "Failed to get recall at precision %f: %s",
            precision_target,
            e,
        )

    # Add the average precision.
    scalars[f"{model_name}_AP"] = pr_calculator.get_average_precision()

  # Add the labels balance score.
  scalars["labels_balance_score"] = calculate_labels_balance_score(
      few_shot_result.labels
  )

  return scalars


def nms(
    embeddings: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Applies Non-Maximum Suppression to bounding boxes.

  Args:
    embeddings: The embeddings of the bounding boxes.
    boxes: The bounding boxes.
    scores: The scores of the bounding boxes.
    iou_threshold: The IoU threshold to use for NMS.

  Returns:
    A tuple containing the filtered embeddings, boxes, and scores.
  """
  if len(boxes) == 0:  # pylint: disable=g-explicit-length-test
    return embeddings, boxes, scores

  # Sort boxes by score in descending order
  order = scores.argsort()[::-1]

  keep_indices = []
  for i in order:
    keep_box = True
    for j in keep_indices:
      if intersection_over_union(boxes[i], boxes[j]) > iou_threshold:
        keep_box = False
        break
    if keep_box:
      keep_indices.append(i)

  return embeddings[keep_indices], boxes[keep_indices], scores[keep_indices]
