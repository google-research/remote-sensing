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

"""API implementation for few-shot retrieval.

This module provides the main entry point for the few-shot retrieval pipeline.
It orchestrates an active learning workflow where a zero-shot model's initial
detections are refined using a small number of user-provided annotations.

Key components:
- `FewShotAlgorithm`: Manages the active learning loop, including example
  selection and model training.
- `FewShotClassifier`: A trained model wrapper used for classifying new, unseen
  embeddings.
"""

from collections.abc import Sequence
import logging

import ml_collections
import numpy as np
from remote_sensing.fewshot import fewshot_api as api
from remote_sensing.fewshot import fewshot_models as models_lib
from remote_sensing.fewshot import sampling
import typing_extensions


class FewShotClassifier(api.AbstractFewShotClassifier):
  """A trained few-shot classifier for performing inference.

  This class wraps a trained `FewShotModel` (e.g., SVM, Logistic Regression)
  and provides a standard interface for classifying new embeddings. It is
  typically created by the `FewShotAlgorithm.train_few_shot_model` method.
  """

  def __init__(self, few_shot_model: models_lib.FewShotModel) -> None:
    """Initializes the model with the state learned during training.

    Args:
      few_shot_model: The internal state (e.g., trained weights, reference
        embeddings, etc.) required for classification. This is provided by the
        FewShotAlgorithm. It is expected to implement two methods: `predict` and
        `score_confidence` that accept embeddings and return classification
        labels and confidence scores respectively.
    """
    self._model = few_shot_model

  @typing_extensions.override
  def classify(self, embeddings: np.ndarray) -> np.ndarray:
    """Classifies new embeddings.

    Args:
      embeddings: The embeddings of the examples to classify. The embeddings are
        expected to be of size (num_examples, embedding_dim).

    Returns:
      The predicted integer labels for the embeddings.
    """
    return self._model.predict(embeddings)

  @typing_extensions.override
  def score_confidence(self, embeddings: np.ndarray) -> np.ndarray:
    """Returns the confidence scores for the embeddings.

    Args:
      embeddings: The embeddings of the examples to score. The embeddings are
        expected to be of size (num_examples, embedding_dim).

    Returns:
      The confidence scores for the embeddings.
    """
    return self._model.score_confidence(embeddings)


class FewShotAlgorithm(api.AbstractFewShotAlgorithm):
  """Builds a `FewShotClassifier` through an active learning workflow.

  This class orchestrates the process of training a few-shot model by
  interactively selecting the most "informative" examples for user labeling.

  The typical workflow is:
  1. Initialize with a configuration.
  2. Use `get_indices_for_classification` to select informative examples for
     labeling.
  3. Use `train_few_shot_model` to train a `FewShotClassifier` using the labeled
     examples.
  """

  def __init__(self, config: ml_collections.ConfigDict) -> None:
    """Initializes the algorithm's internal state."""
    self._config = config

  def _get_sampler(self) -> sampling.BaseSampler:
    """Initializes the few-shot sampler."""
    if self._config["sampler"]:
      sampler_config = self._config.sampler
      sampler_name = sampler_config.get("name")
      if not sampler_name:
        raise ValueError("Sampler name not provided in config.")

      if not hasattr(sampling, sampler_name):
        raise ValueError(f"Unknown sampling algorithm: {sampler_name}")

      sampler_params = sampler_config.get("params", {})
      logging.info(
          "Using sampler %s with params %s", sampler_name, sampler_params
      )
      return getattr(sampling, sampler_name)(**sampler_params)
    else:
      logging.info(
          "No sampler config found. Using the default BaselineSampler."
      )
      return sampling.BaselineSampler()

  def _get_model(self) -> models_lib.FewShotModel:
    """Initializes the few-shot model based on the config."""
    if self._config.get("few_shot_model"):
      model_config = self._config.few_shot_model
      model_name = model_config.get("name")
      if not model_name:
        raise ValueError("Few-shot model name not provided in config.")
      if not hasattr(models_lib, model_name):
        raise ValueError(f"Unknown few-shot model: {model_name}")

      model_params = model_config.get("params", {})
      logging.info(
          "Using few-shot model %s with params %s",
          model_name,
          model_params,
      )
      return getattr(models_lib, model_name)(**model_params)
    else:
      logging.info("No model config found. Using the default SVMModel.")
      return models_lib.SVMModel()

  @typing_extensions.override
  def get_indices_for_annotation(
      self,
      zero_shot_examples: Sequence[api.ZeroShotExample],
      number_of_samples: int,
  ) -> Sequence[int]:
    """Selects the most informative examples that need user labels.

    This is the core of the "active learning" loop. The method uses a sampling
    strategy (e.g., uncertainty sampling, diversity sampling) to select the best
    examples to present to the user for labeling.

    Args:
      zero_shot_examples: A sequence of candidate examples.
      number_of_samples: The desired number of examples to be labeled.

    Returns:
      A sequence of integer indices corresponding to the examples in the
      `zero_shot_examples` list that should be presented to the user.
    """

    sampler = self._get_sampler()
    return sampler.sample(
        embeddings=np.array(
            [example.embedding for example in zero_shot_examples]
        ),
        confidences=np.array([example.score for example in zero_shot_examples]),
        num_samples=number_of_samples,
    )

  @typing_extensions.override
  def train_few_shot_model(
      self, annotated_examples: Sequence[api.ZeroShotExample]
  ) -> api.AbstractFewShotClassifier:
    """Trains and returns a final `FewShotClassifier`.

    This method uses the provided annotated examples (with labels) to train
    the classification model.

    Args:
      annotated_examples: A sequence of examples with user-provided labels.

    Returns:
      A trained `FewShotClassifier` instance ready for inference.

    Raises:
      ValueError: If no user labels have been provided.
    """

    embeddings = []
    labels = []
    for example in annotated_examples:
      if example.label is not None:
        embeddings.append(example.embedding)
        labels.append(example.label)

    if not embeddings:
      raise ValueError(
          "No annotated examples with labels provided for training."
      )

    model = self._get_model()
    model.train(
        train_data=np.array(embeddings),
        train_labels=np.array(labels),
    )
    return FewShotClassifier(model)
