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

"""Test the utilities module.."""

import numpy as np
import pytest
from remote_sensing.fewshot import utils
from sklearn import metrics


class TestUtils:

  def test_crop_with_margin(self):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[60:80, 60:80] = 1  # Fill a region with 1
    bbox = (0.4, 0.4, 0.6, 0.6)  # Normalized coordinates
    margin_ratio = 0.5
    cropped_image = utils.crop_with_margin(image, bbox, margin_ratio)
    assert cropped_image.shape == (40, 40, 3)
    assert np.sum(cropped_image) == 3 * 10 * 10

    bbox = (0.0, 0.0, 1.0, 1.0)  # Normalized coordinates
    margin_ratio = 0.1
    cropped_image = utils.crop_with_margin(image, bbox, margin_ratio)
    assert cropped_image.shape == (100, 100, 3)

  def test_crop_with_margin_max_area(self):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (0.1, 0.1, 0.9, 0.9)  # Bounding box covering most of the image

    # Test case 1: max_crop_area is smaller than the crop
    max_area = 50 * 50
    cropped_image = utils.crop_with_margin(
        image, bbox, margin_ratio=0, max_crop_area=max_area
    )
    cropped_area = cropped_image.shape[0] * cropped_image.shape[1]
    assert cropped_area <= max_area
    assert cropped_image.shape[1] / cropped_image.shape[0] == pytest.approx(
        1.0, abs=0.01
    )

    # Test case 2: max_crop_area is larger than the crop
    max_area = 100 * 100
    cropped_image = utils.crop_with_margin(
        image, bbox, margin_ratio=0, max_crop_area=max_area
    )
    expected_size = int(100 * (0.9 - 0.1))  # Expected size without resizing
    assert cropped_image.shape[0] == expected_size
    assert cropped_image.shape[1] == expected_size

    # Test case 3: max_crop_area is None
    cropped_image = utils.crop_with_margin(
        image, bbox, margin_ratio=0, max_crop_area=None
    )
    expected_size = int(100 * (0.9 - 0.1))  # Expected size without resizing
    assert cropped_image.shape[0] == expected_size
    assert cropped_image.shape[1] == expected_size

  def test_intersection_over_union(self):
    # Overlapping boxes
    box1 = np.array([0, 0, 2, 2])
    box2 = np.array([1, 1, 3, 3])
    iou = utils.intersection_over_union(box1, box2)
    assert iou == pytest.approx(1.0 / 7.0)

    # Non-overlapping boxes
    box1 = np.array([0, 0, 1, 1])
    box2 = np.array([2, 2, 3, 3])
    iou = utils.intersection_over_union(box1, box2)
    assert iou == 0.0

    # Identical boxes
    box1 = np.array([0, 0, 2, 2])
    box2 = np.array([0, 0, 2, 2])
    iou = utils.intersection_over_union(box1, box2)
    assert iou == 1.0

  def test_get_recall_at_precision_achievable(self):
    y_true = np.array([0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8])
    false_negatives_count = 0
    target_precision = 0.8
    pr_calculator = utils.PrecisionAtRecallCalculator(
        y_true=y_true,
        y_scores=y_scores,
        false_negatives_count=false_negatives_count,
    )
    recall, precision, threshold = pr_calculator.get_recall_at_precision(
        target_precision
    )
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert threshold == pytest.approx(0.2)

  def test_get_recall_at_precision_not_achievable(self):
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.3, 0.4])
    false_negatives_count = 0
    target_precision = 0.8
    with pytest.raises(
        ValueError, match='Target precision 0.8 achieved at recall 0.'
    ):
      pr_calculator = utils.PrecisionAtRecallCalculator(
          y_true=y_true,
          y_scores=y_scores,
          false_negatives_count=false_negatives_count,
      )
      pr_calculator.get_recall_at_precision(target_precision)

  def test_no_positives(self):
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.4, 0.3, 0.4])
    false_negatives_count = 0
    target_precision = 0.8
    with pytest.raises(
        ValueError,
        match='No positives in dataset. Target precision not calculated.',
    ):
      utils.PrecisionAtRecallCalculator(
          y_true=y_true,
          y_scores=y_scores,
          false_negatives_count=false_negatives_count,
      ).get_recall_at_precision(target_precision)

  def test_classification_report_simple(self):
    zero_shot_result = utils.ModelResults(
        scores=np.array([0.5, 0.7, 0.3, 0.09]),
        predictions=np.array([1, 1, 1, 1]),
        labels=np.array([1, 0, 1, 0]),
        false_negatives_count=2,
    )
    few_shot_result = utils.ModelResults(
        scores=np.array([0.8, 0.6, 0.9, 0.2]),
        predictions=np.array([1, 1, 1, 0]),
        labels=np.array([1, 0, 1, 0]),
        false_negatives_count=2,
    )
    report = utils.classification_report(few_shot_result, zero_shot_result)

    expected_report = {
        'ZS_false_negatives_count': 2,
        'ZS_recall_0': 0.0,
        'ZS_precision_0': 0.0,
        'ZS_support_0': 2.0,
        'ZS_f1_score_0': 0.0,
        'ZS_recall': 0.5,
        'ZS_precision': 0.5,
        'ZS_support': 4.0,
        'ZS_f1_score': 0.5,
        'ZS_accuracy': 0.3333333333333333,
        'ZS_R@P0.5': 0.5,
        'ZS_P@P0.5': 0.5,
        'ZS_T@P0.5': 0.09,
        'ZS_R@P0.6': 0.5,
        'ZS_P@P0.6': 0.6666666666666666,
        'ZS_T@P0.6': 0.09,
        'ZS_AP': 0.29166666666666663,
        'FS_false_negatives_count': 2,
        'FS_recall_0': 0.5,
        'FS_precision_0': 0.3333333333333333,
        'FS_support_0': 2.0,
        'FS_f1_score_0': 0.4,
        'FS_recall': 0.5,
        'FS_precision': 0.6666666666666666,
        'FS_support': 4.0,
        'FS_f1_score': 0.5714285714285714,
        'FS_accuracy': 0.5,
        'FS_R@P0.5': 0.5,
        'FS_P@P0.5': 0.5,
        'FS_T@P0.5': 0.2,
        'FS_R@P0.6': 0.5,
        'FS_P@P0.6': 0.6666666666666666,
        'FS_T@P0.6': 0.2,
        'FS_R@P0.7': 0.5,
        'FS_P@P0.7': 1.0,
        'FS_T@P0.7': 0.6,
        'FS_R@P0.8': 0.5,
        'FS_P@P0.8': 1.0,
        'FS_T@P0.8': 0.6,
        'FS_R@P0.9': 0.5,
        'FS_P@P0.9': 1.0,
        'FS_T@P0.9': 0.6,
        'FS_AP': 0.5,
        'labels_balance_score': 1.0,
    }

    assert report == expected_report

  def test_nms_empty_boxes(self):
    embeddings = np.array([])
    boxes = np.array([])
    scores = np.array([])
    filtered_embeddings, filtered_boxes, filtered_scores = utils.nms(
        embeddings, boxes, scores
    )
    assert len(filtered_embeddings) == 0  # pylint: disable=g-explicit-length-test
    assert len(filtered_boxes) == 0  # pylint: disable=g-explicit-length-test
    assert len(filtered_scores) == 0  # pylint: disable=g-explicit-length-test

  def test_nms_single_box(self):
    embeddings = np.array([[1, 2]])
    boxes = np.array([[0, 0, 1, 1]])
    scores = np.array([0.9])
    filtered_embeddings, filtered_boxes, filtered_scores = utils.nms(
        embeddings, boxes, scores
    )
    np.testing.assert_array_equal(filtered_embeddings, embeddings)
    np.testing.assert_array_equal(filtered_boxes, boxes)
    np.testing.assert_array_equal(filtered_scores, scores)

  def test_nms_overlapping_boxes(self):
    embeddings = np.array([[1, 2], [3, 4], [5, 6]])
    boxes = np.array([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1], [2, 2, 3, 3]])
    scores = np.array([0.9, 0.8, 0.7])
    filtered_embeddings, filtered_boxes, filtered_scores = utils.nms(
        embeddings, boxes, scores, iou_threshold=0.5
    )
    np.testing.assert_array_equal(
        filtered_embeddings, np.array([[1, 2], [5, 6]])
    )
    np.testing.assert_array_equal(
        filtered_boxes, np.array([[0, 0, 1, 1], [2, 2, 3, 3]])
    )
    np.testing.assert_array_equal(filtered_scores, np.array([0.9, 0.7]))

  def test_nms_non_overlapping_boxes(self):
    embeddings = np.array([[1, 2], [3, 4], [5, 6]])
    boxes = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]])
    scores = np.array([0.9, 0.8, 0.7])
    filtered_embeddings, filtered_boxes, filtered_scores = utils.nms(
        embeddings, boxes, scores, iou_threshold=0.5
    )
    np.testing.assert_array_equal(filtered_embeddings, embeddings)
    np.testing.assert_array_equal(filtered_boxes, boxes)
    np.testing.assert_array_equal(filtered_scores, scores)

  @pytest.mark.parametrize(
      'y_true,y_scores,false_negatives_count,expected_ap',
      [
          (
              np.array([0, 1, 0, 1]),
              np.array([0.1, 0.4, 0.35, 0.8]),
              0,
              1.0,
          ),
          (
              np.array([0, 0, 1, 1]),
              np.array([0.1, 0.4, 0.35, 0.2]),
              0,
              0.5833333333333333,
          ),
          (
              np.array([0, 0, 1, 1]),
              np.array([0.1, 0.4, 0.35, 0.2]),
              1,
              0.38888888888888884,
          ),
      ],
      ids=['perfect', 'good', 'with_false_negative'],
  )
  def test_get_average_precision(
      self, y_true, y_scores, false_negatives_count, expected_ap
  ):
    calculator = utils.PrecisionAtRecallCalculator(
        y_true, y_scores, false_negatives_count
    )
    ap = calculator.get_average_precision()
    if expected_ap is not None:
      assert ap == pytest.approx(expected_ap, abs=1e-5)
    else:
      precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
      expected_ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
      assert ap == pytest.approx(expected_ap, abs=1e-5)

  @pytest.mark.parametrize(
      'labels,expected_score',
      [
          (np.array([0, 1, 0, 1]), 1.0),
          (np.array([0, 0, 0, 0]), 0.0),
          (np.array([1, 1, 1, 1]), 0.0),
          (np.array([0, 1, 0, 0]), 0.5),
          (np.array([]), 0.0),
      ],
      ids=[
          'perfect_balance',
          'no_balance_all_zeros',
          'no_balance_all_ones',
          'some_balance',
          'empty_array',
      ],
  )
  def test_calculate_labels_balance_score(self, labels, expected_score):
    score = utils.calculate_labels_balance_score(labels)
    assert score == expected_score
