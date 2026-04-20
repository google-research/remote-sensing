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

"""Tests for `sampling` module of the few-shot retrieval pipeline."""

import numpy as np
import pytest
from remote_sensing.fewshot import sampling


class TestBaselineSampler:

  @pytest.mark.parametrize(
      "embeddings,confidences,num_samples,expected_indices,expected_error",
      [
          (
              np.array([[1, 2], [3, 4]]),
              np.array([0.1, 0.9]),
              3,
              None,
              ValueError,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6]]),
              np.array([0.1, 0.9, 0.5]),
              2,
              [0, 2],
              None,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
              np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.3]),
              3,
              [0, 5, 3],
              None,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6]]),
              np.array([0.5, 0.5, 0.5]),
              2,
              [2, 0],  # tie break is based on order
              None,
          ),
          (
              np.array([[1, 2], [3, 4]]),
              np.array([0.1, 0.9]),
              2,
              [0, 1],
              None,
          ),
      ],
      ids=[
          "not_enough_embeddings",
          "small_pool",
          "larger_pool",
          "all_same_confidence",
          "num_embeddings_smaller_than_initial_uncertain_pool",
      ],
  )
  def test_baseline_sampler(
      self,
      embeddings,
      confidences,
      num_samples,
      expected_indices,
      expected_error,
  ):
    sampler = sampling.BaselineSampler(initial_uncertain_pool_factor=2)
    if expected_error:
      with pytest.raises(expected_error):
        sampler.sample(embeddings, confidences, num_samples)
    else:
      indices = sampler.sample(embeddings, confidences, num_samples)
      np.testing.assert_array_equal(indices, expected_indices)

  def test_baseline_sampler_initial_uncertain_pool_factor(self):
    embeddings = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    confidences = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    num_samples = 2

    # Test with a small initial_uncertain_pool_factor
    sampler_small_factor = sampling.BaselineSampler(
        initial_uncertain_pool_factor=1
    )
    indices_small_factor = sampler_small_factor.sample(
        embeddings, confidences, num_samples
    )
    assert len(indices_small_factor) == num_samples
    assert 0 in indices_small_factor  # highest uncertainty is selected
    np.testing.assert_array_equal(
        sampler_small_factor.get_selected_indices(), np.array([0, 3])
    )

    # Test with a large initial_uncertain_pool_factor
    sampler_large_factor = sampling.BaselineSampler(
        initial_uncertain_pool_factor=4
    )
    indices_large_factor = sampler_large_factor.sample(
        embeddings, confidences, num_samples
    )
    assert len(indices_large_factor) == num_samples
    assert 0 in indices_large_factor  # highest uncertainty is selected
    np.testing.assert_array_equal(
        sampler_large_factor.get_selected_indices(), np.array([0, 4])
    )


class TestKlusteredMarginalEmbeddingsBasedSampler:

  @pytest.mark.parametrize(
      "embeddings,confidences,num_samples,candidate_frac,expected_indices,expected_error,filtering_method",
      [
          (
              np.array([[1, 2], [3, 4]]),
              np.array([0.1, 0.9]),
              3,
              1,
              None,
              ValueError,
              sampling.FilteringMethod.PERCENTILE,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6]]),
              np.array([0.1, 0.9, 0.5]),
              2,
              1,
              [0, 2],
              None,
              sampling.FilteringMethod.PERCENTILE,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
              np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.3]),
              3,
              0.6,
              [2, 5, 3],
              None,
              sampling.FilteringMethod.PERCENTILE,
          ),
          (
              np.array([[1, 2], [3, 4], [5, 6]]),
              np.array([0.5, 0.5, 0.5]),
              2,
              1,
              [2, 0],  # tie break is based on order
              None,
              sampling.FilteringMethod.PERCENTILE,
          ),
      ],
      ids=[
          "not_enough_embeddings",
          "small_pool",
          "larger_pool",
          "all_same_confidence",
      ],
  )
  def test_klustered_marginal_embeddings_based_sampler(
      self,
      embeddings,
      confidences,
      num_samples,
      candidate_frac,
      expected_indices,
      expected_error,
      filtering_method,
  ):
    sampler = sampling.ClusteredMarginalEmbeddingsBasedSampler(
        candidate_frac=candidate_frac,
        filtering_method=filtering_method,
    )

    if expected_error:
      with pytest.raises(expected_error):
        sampler.sample(embeddings, confidences, num_samples)
    else:
      indices = sampler.sample(embeddings, confidences, num_samples)
      np.testing.assert_array_equal(np.sort(indices), np.sort(expected_indices))

  def test_mismatched_lengths(self):
    # Verifies that the function handles a mismatch between the number of
    # embeddings and similarity scores. Expects a ValueError due to length
    # discrepancy.
    embeddings = np.array([[1, 1], [2, 2]])
    scores = np.array([0.9])
    sampler = sampling.ClusteredMarginalEmbeddingsBasedSampler(
        candidate_frac=1,
    )
    with pytest.raises(ValueError):
      sampler.sample(embeddings, scores, num_samples=2)

  def test_num_samples_equal_to_candidates_from_top(self):
    # Checks the scenario where the requested number of samples is exactly equal
    # to the number of candidates determined by `candidate_frac`. Expects all
    # available candidates to be returned.
    embeddings = np.array(
        [
            [1, 1],
            [1.1, 1.1],
            [1.2, 1.2],
            [5, 5],
            [5.1, 5.1],
            [5.2, 5.2],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.89, 0.88, 0.7, 0.69, 0.68], dtype=np.float32)
    num_samples = 4
    # All 6 are candidates
    sampler = sampling.ClusteredMarginalEmbeddingsBasedSampler(
        candidate_frac=1.0,
    )
    selected_indices = sampler.sample(embeddings, scores, num_samples)

    assert selected_indices.shape == (num_samples,)
