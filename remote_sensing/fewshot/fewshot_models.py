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

"""Library that implements different models for the few-shot retrieval."""

import abc

import numpy as np
from overrides import overrides
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm


class FewShotModel(abc.ABC):
  """Abstract base class for few-shot models."""

  def __init__(self, **kwargs):
    """Initializes the model."""
    self.model_params = kwargs

  @abc.abstractmethod
  def train(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
    """Trains the model.

    Args:
      train_data: The training data.
      train_labels: The training labels.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self, test_data: np.ndarray) -> np.ndarray:
    """Predicts the labels for the test data.

    Args:
      test_data: The test data.

    Returns:
      The predicted labels.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def score_confidence(self, test_data: np.ndarray) -> np.ndarray:
    """Returns the confidence scores for the test data.

    Args:
      test_data: The test data.

    Returns:
      The confidence scores.
    """
    raise NotImplementedError()


class SVMModel(FewShotModel):
  """A few-shot model based on sklearn SVM."""

  def __init__(self, *, kernel="rbf", C=10.0, gamma="scale", **kwargs):
    super().__init__(kernel=kernel, C=C, gamma=gamma, **kwargs)
    self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)

  @overrides
  def train(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
    self.model.fit(train_data, train_labels)

  @overrides
  def predict(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.predict(test_data)

  @overrides
  def score_confidence(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.decision_function(test_data)


class LogisticRegressionModel(FewShotModel):
  """A few-shot model based on sklearn Logistic Regression."""

  def __init__(self, *, C=1.0, **kwargs):
    super().__init__(C=C, **kwargs)
    self.model = linear_model.LogisticRegression(C=C)

  @overrides
  def train(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
    self.model.fit(train_data, train_labels)

  @overrides
  def predict(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.predict(test_data)

  @overrides
  def score_confidence(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.predict_proba(test_data)[:, 1]


class RandomForestModel(FewShotModel):
  """A few-shot model based on sklearn Random Forest."""

  def __init__(
      self, *, n_estimators=100, max_depth=None, random_state=42, **kwargs
  ):
    super().__init__(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs,
    )
    self.model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

  @overrides
  def train(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
    self.model.fit(train_data, train_labels)

  @overrides
  def predict(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.predict(test_data)

  @overrides
  def score_confidence(self, test_data: np.ndarray) -> np.ndarray:
    return self.model.predict_proba(test_data)[:, 1]
