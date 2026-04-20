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

"""Test the few-shot retrieval API and algorithms."""

from unittest import mock

import ml_collections
import numpy as np
import pytest
from remote_sensing.fewshot import algorithms
from remote_sensing.fewshot import fewshot_models as models_lib
from remote_sensing.fewshot import fewshot_ovd_api as api
from remote_sensing.fewshot import sampling


class TestApi:

  def test_few_shot_model(self):
    mock_internal_model = mock.Mock()
    mock_internal_model.predict.return_value = 1
    model = algorithms.FewShotClassifier(mock_internal_model)
    embedding = np.array([0.5, 0.6])
    prediction = model.classify(embedding)
    mock_internal_model.predict.assert_called_once_with(embedding)
    assert prediction == 1

  def test_get_indices_for_annotation(self):
    # 1. Create a mock sampler.
    mock_sampler = mock.create_autospec(sampling.BaseSampler, instance=True)
    mock_sampler.sample.return_value = [0, 2]

    # 2. Create a config.
    config = ml_collections.ConfigDict()

    # 3. Instantiate the algorithm.
    algorithm = algorithms.FewShotAlgorithm(config)

    # 4. Patch the _get_sampler method to return the mock sampler.
    with mock.patch.object(
        algorithm, "_get_sampler", return_value=mock_sampler
    ) as mock_get_sampler:
      # 5. Create some dummy ZeroShotExamples.
      examples = [
          api.ZeroShotExample(embedding=np.array([0.1, 0.2]), score=0.9),
          api.ZeroShotExample(embedding=np.array([0.3, 0.4]), score=0.8),
          api.ZeroShotExample(embedding=np.array([0.5, 0.6]), score=0.7),
      ]
      num_samples = 2

      # 6. Call the method under test.
      indices = algorithm.get_indices_for_annotation(examples, num_samples)

      # 7. Assertions.
      assert indices == [0, 2]
      mock_get_sampler.assert_called_once()

      # Check that the sampler's sample method was called with the correct
      # arguments.
      mock_sampler.sample.assert_called_once()
      _, kwargs = mock_sampler.sample.call_args
      np.testing.assert_array_equal(
          kwargs["embeddings"], np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
      )
      np.testing.assert_array_equal(
          kwargs["confidences"], np.array([0.9, 0.8, 0.7])
      )
      assert kwargs["num_samples"] == num_samples

  def test_train_few_shot_model(self):
    # 1. Create a mock for the internal model.
    mock_internal_model = mock.create_autospec(
        models_lib.FewShotModel, instance=True
    )

    # 2. Create a config.
    config = ml_collections.ConfigDict()

    # 3. Instantiate the algorithm.
    algorithm = algorithms.FewShotAlgorithm(config)

    # 4. Patch the _get_model method to return the mock model.
    with mock.patch.object(
        algorithm, "_get_model", return_value=mock_internal_model
    ) as mock_get_model:
      # 5. Create some dummy annotated ZeroShotExamples.
      annotated_examples = [
          api.ZeroShotExample(
              embedding=np.array([0.1, 0.2]), score=0.9, label=1
          ),
          api.ZeroShotExample(
              embedding=np.array([0.3, 0.4]), score=0.8, label=0
          ),
          api.ZeroShotExample(
              embedding=np.array([0.5, 0.6]), score=0.7, label=1
          ),
      ]

      # 6. Call the method under test.
      few_shot_model = algorithm.train_few_shot_model(annotated_examples)

      # 7. Assertions.
      assert isinstance(few_shot_model, algorithms.FewShotClassifier)
      mock_get_model.assert_called_once()

      # Check that the internal model's train method was called with the correct
      # arguments.
      mock_internal_model.train.assert_called_once()
      _, kwargs = mock_internal_model.train.call_args
      np.testing.assert_array_equal(
          kwargs["train_data"], np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
      )
      np.testing.assert_array_equal(kwargs["train_labels"], np.array([1, 0, 1]))

  def test_train_few_shot_model_no_labels(self):
    # 1. Create a config.
    config = ml_collections.ConfigDict()

    # 2. Instantiate the algorithm.
    algorithm = algorithms.FewShotAlgorithm(config)

    # 3. Create examples with no labels.
    annotated_examples = [
        api.ZeroShotExample(embedding=np.array([0.1, 0.2]), score=0.9),
        api.ZeroShotExample(embedding=np.array([0.3, 0.4]), score=0.8),
    ]

    # 4. Check that ValueError is raised.
    with pytest.raises(ValueError, match="No annotated examples with labels"):
      algorithm.train_few_shot_model(annotated_examples)
