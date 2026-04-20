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

"""Sampling methods to extract (few) samples for training a few-shot model.

The selected samples are annotated by the user and therefore few and it is
important that the sampling method will select samples that are diverse and
representative of the data.
"""

import abc
import enum
import logging
from typing import Optional

import numpy as np
from overrides import overrides
from scipy import ndimage
from sklearn import cluster
from sklearn import decomposition
from sklearn import neighbors
from sklearn.metrics import pairwise

PERCENTS = 100
EPSILON_FOR_CANDIDATE_FRACTION = 0.05


def select_diversified_candidates(
    embeddings: np.ndarray,
    similarity_scores: np.ndarray,
    num_samples: int,
    candidate_frac: float = 0.1,
    max_num_samples_to_process: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
  """Selects a diverse subset of candidates using a two-stage process.

  First, it filters for top candidates based on similarity scores. Then, it
  uses K-Means clustering on their embeddings to find representative samples,
  selecting the one closest to each cluster's centroid.

  Args:
      embeddings: A 2D numpy array of item embeddings.
      similarity_scores: A 1D numpy array of similarity scores for each item.
      num_samples: The number of diverse samples to select.
      candidate_frac: Fraction of top-scoring items to consider as candidates.
      max_num_samples_to_process: The maximum number of candidates to process
        with K-Means.

  Returns:
      A tuple containing:
      - A numpy array of the selected embeddings.
      - A numpy array of the original indices of the selected items.
  """
  # Input Validation
  if num_samples <= 0:
    raise ValueError("num_samples must be a positive integer.")

  num_embeddings = len(embeddings)
  if num_embeddings == 0:
    raise ValueError("Embeddings array cannot be empty.")

  if num_embeddings != len(similarity_scores):
    raise ValueError(
        f"Mismatch in lengths of embeddings ({num_embeddings}) and "
        f"scores ({len(similarity_scores)})."
    )

  scores = np.squeeze(similarity_scores)
  if scores.ndim != 1:
    raise ValueError(
        "similarity_scores must be a 1D array, but has shape "
        f"{similarity_scores.shape} after squeeze."
    )

  # Handle Edge Cases
  # If only one sample is needed, return the one with the highest score.
  if num_samples == 1:
    best_idx = np.argmax(scores)
    return embeddings[best_idx : best_idx + 1], np.array([best_idx])

  ## Select Top Candidates for Clustering
  # Determine the number of top candidates to consider.
  num_candidates = min(
      int(np.ceil(candidate_frac * num_embeddings)),
      max_num_samples_to_process,
  )

  if num_candidates < num_samples:
    raise ValueError(
        f"Not enough candidates ({num_candidates}) to select {num_samples} "
        "diverse samples. Consider increasing candidate_frac or "
        "max_num_samples_to_process."
    )

  # Get indices of candidates with the highest scores.
  # np.argsort sorts ascending, so we take the last `num_candidates`.
  candidate_indices = np.argsort(scores)[-num_candidates:]
  candidate_embeddings = embeddings[candidate_indices]

  # If the number of candidates is exactly what we need, return them.
  if num_candidates == num_samples:
    # Return candidates ordered by score (descending) for consistency.
    return candidate_embeddings[::-1], candidate_indices[::-1]

  # Diversify Candidates with K-Means
  kmeans = cluster.KMeans(
      n_clusters=num_samples, init="k-means++", random_state=0, n_init="auto"
  )
  labels = kmeans.fit_predict(candidate_embeddings)
  centroids = kmeans.cluster_centers_

  closest_candidate_indices = []
  for i in range(num_samples):
    # Identify all candidates belonging to the current cluster.
    cluster_mask = labels == i
    indices_in_cluster = np.where(cluster_mask)[0]

    # Find the candidate closest to the centroid of this cluster.
    distances = np.linalg.norm(
        candidate_embeddings[cluster_mask] - centroids[i], axis=1
    )
    closest_idx_in_cluster = np.argmin(distances)

    # Get the index of the chosen sample within `candidate_embeddings`.
    closest_candidate_idx = indices_in_cluster[closest_idx_in_cluster]
    closest_candidate_indices.append(closest_candidate_idx)

  ## Map Indices and Return Results
  # Map indices from the candidate set back to the original full set.
  final_indices = candidate_indices[closest_candidate_indices]
  final_embeddings = embeddings[final_indices]

  return final_embeddings, final_indices


class BaseSampler(abc.ABC):
  """Abstract base class for sampling methods."""

  def __init__(self, **kwargs):
    """Initializes the sampler with custom parameters."""
    self.params = kwargs
    self._selected_indices: Optional[np.ndarray] = None

  @abc.abstractmethod
  def sample(
      self, embeddings: np.ndarray, confidences: np.ndarray, num_samples: int
  ) -> np.ndarray:
    """Selects a subset of embeddings and scores for annotation.

    Args:
      embeddings: A NumPy array of shape (num_examples, embedding_dim)
        representing the embeddings of the unlabeled data.
      confidences: A NumPy array of shape (num_examples,) representing the
        previous model's (possibly zero-shot) confidence for each example (0 to
        1).
      num_samples: The desired number of examples to select (N).

    Returns:
      A np.array of indices of the selected embeddings and scores.
    """
    raise NotImplementedError()

  def get_selected_indices(self) -> Optional[np.ndarray]:
    """Returns the indices of the selected samples."""
    if self._selected_indices is None:
      raise ValueError("No samples were selected yet.")
    return self._selected_indices


class BaselineSampler(BaseSampler):
  """Baseline sampling method based on uncertainty and diversity.

  Attributes:
    initial_uncertain_pool_factor: The factor by which to multiply the number of
      samples to determine the size of the initial uncertain pool.
  """

  def __init__(self, *, initial_uncertain_pool_factor: int = 3, **kwargs):
    """Initializes the sampler with custom parameters."""
    super().__init__(
        initial_uncertain_pool_factor=initial_uncertain_pool_factor,
        **kwargs,
    )
    self.initial_uncertain_pool_factor = initial_uncertain_pool_factor

  @overrides
  def sample(
      self,
      embeddings: np.ndarray,
      confidences: np.ndarray,
      num_samples: int,
  ) -> np.ndarray:
    """Run the algorithm to select the best samples for annotation.

    Args:
      embeddings: A NumPy array of shape (num_examples, embedding_dim)
        representing the embeddings of the unlabeled data.
      confidences: A NumPy array of shape (num_examples,) representing the
        previous model's (possibly zero-shot) confidence for each example (0 to
        1).
      num_samples: The desired number of examples to select (N).

    Returns:
        A NumPy array containing the indices of the selected examples.
    """

    num_total_embeddings = embeddings.shape[0]

    if num_total_embeddings < num_samples:
      raise ValueError(
          f"Not enough total embeddings ({num_total_embeddings}) to select "
          f"{num_samples} samples."
      )

    # Calculate Uncertainty Scores
    # Lower confidence means higher uncertainty
    uncertainty_scores = 1 - confidences

    # Uncertainty Filtering: Select a larger pool of most uncertain examples
    # Sort indices by uncertainty in descending order
    sorted_indices_by_uncertainty = np.argsort(uncertainty_scores)[::-1]

    # Determine the size of the initial uncertain pool
    num_uncertain_pool = min(
        num_total_embeddings,
        num_samples * self.initial_uncertain_pool_factor,
    )

    # If num_total_embeddings is very small, ensure num_uncertain_pool is at
    # least num_samples
    if num_uncertain_pool < num_samples:
      num_uncertain_pool = num_samples
      logging.warning(
          "Not enough total embeddings for initial_uncertain_pool_factor."
          " Adjusting pool size to %d.",
          num_uncertain_pool,
      )

    uncertain_pool_indices = sorted_indices_by_uncertainty[:num_uncertain_pool]
    uncertain_pool_embeddings = embeddings[uncertain_pool_indices]

    # Diversity Selection (Farthest-First Traversal on the uncertain pool)
    selected_indices_in_pool = []

    if num_uncertain_pool == 0:
      return np.array([])  # No examples to select from

    # Start with the most uncertain example from the pool
    # This also acts as a diversity anchor for the farthest-first traversal
    first_selection_idx_in_pool = 0
    selected_indices_in_pool.append(first_selection_idx_in_pool)

    # Calculate distances from the first selected point to all other points in
    # the pool using squared Euclidean distance for efficiency, as it preserves
    # ordering.
    distances = pairwise.euclidean_distances(
        uncertain_pool_embeddings[first_selection_idx_in_pool].reshape(1, -1),
        uncertain_pool_embeddings,
    ).flatten()

    for _ in range(1, num_samples):
      if len(selected_indices_in_pool) >= num_uncertain_pool:
        break  # Cannot select more samples than available in the pool

      # Find the point in the pool that is farthest from any already selected
      # point.
      # Update distances: for each unselected point, its distance is the minimum
      # distance to any of the currently selected points.
      farthest_idx_in_pool = np.argmax(distances)
      selected_indices_in_pool.append(farthest_idx_in_pool)

      # Update distances for subsequent iterations
      if (
          len(selected_indices_in_pool) < num_samples
      ):  # No need to update if we've already selected enough
        new_distances_from_last_selected = pairwise.euclidean_distances(
            uncertain_pool_embeddings[farthest_idx_in_pool].reshape(1, -1),
            uncertain_pool_embeddings,
        ).flatten()
        distances = np.minimum(distances, new_distances_from_last_selected)

    # Map back to original indices
    self._selected_indices = uncertain_pool_indices[selected_indices_in_pool]
    return self._selected_indices


class FilteringMethod(enum.Enum):
  PERCENTILE = enum.auto()
  GRADIENT_TILT_BASED = enum.auto()
  KDE = enum.auto()


class ClusteredMarginalEmbeddingsBasedSampler(BaseSampler):
  """Two stages K-clustered marginal embeddings sampling method.

  selecting diversed representative small amount of samples from the uncertain
  pool.

  First, it filters out high-scoring samples that are far from the decision
  boundary of the target class. Second, it identifies a high-relevance subset
  by filtering the initial image_embeddings to retain only the top candidates
  based on their similarity_scores, with the size of this subset determined
  by candidate_frac. Third, to ensure diversity within this high-scoring
  group, it applies K-Means clustering to the candidate embeddings, setting
  the number of clusters equal to num_shots. This partitions the candidates
  into distinct groups based on their feature similarity. The final output is
  generated by selecting the single most representative example from each
  cluster, specifically the one whose embedding is closest to the cluster's
  centroid, thus yielding a final set that is both highly relevant to the
  target and varied in its representation.
  """

  def __init__(
      self,
      candidate_frac: float = 0.1,
      filtering_method: FilteringMethod = FilteringMethod.KDE,
      filtering_method_threshold: float = 0.7,
      positive_samples_imbalance_ratio: float = 0.5,
      sigma_gaussian_filter_score_hist_smoothing: float = 2.0,
      max_num_samples_to_process: int = 500,
      kde_bandwidth: float = 0.07,
      density_threshold_ratio: float = 0.2,
      margin_width: float = 0.05,
  ):
    """Initializes the sampler with custom parameters.

    Args:
      candidate_frac: The fraction of top-scoring images to consider as
        candidates for diversity selection.
      filtering_method: The method to use for filtering marginal candidates.
      filtering_method_threshold: A threshold  used bythe filtering method.
      positive_samples_imbalance_ratio: The threshold to decide if to use SMOTE
        or SVMSMOTE, in post processing.
      sigma_gaussian_filter_score_hist_smoothing: The sigma of the gaussian
        filter used for smoothing the score histogram.
      max_num_samples_to_process: The maximum number of samples to process in
        the K-means algorithm.
      kde_bandwidth: The bandwidth of the Gaussian kernel used in the Kernel
        Density Estimation.
      density_threshold_ratio: The ratio of the peak density to use as a
        threshold in the Kernel Density Estimation.
      margin_width: The margin width to expand the area near the thresholds in
        the Kernel Density Estimation.
    """

    # Set the filtering method based on the provided name.
    if filtering_method == FilteringMethod.PERCENTILE:
      self.filtering_method = self._filter_candidates_by_percentile
    elif filtering_method == FilteringMethod.GRADIENT_TILT_BASED:
      self.filtering_method = self._filter_candidates_by_gradient_tilt
    elif filtering_method == FilteringMethod.KDE:
      self.filtering_method = self._filter_candidates_by_gkde
    else:
      # This is the default case, which handles any other value.
      raise ValueError("Invalid filtering method.")

    if not (0 < candidate_frac <= 1):  # Validate inputs
      raise ValueError("candidate_frac must be in (0 and 1] (exclusive).")

    self.candidate_frac = candidate_frac
    self.marginal_candidates_threshold = filtering_method_threshold
    self.positive_samples_imbalance_ratio = positive_samples_imbalance_ratio
    self.sigma_gaussian_filter_score_hist_smoothing = (
        sigma_gaussian_filter_score_hist_smoothing
    )
    self.max_num_samples_to_process = max_num_samples_to_process
    self.kde_bandwidth = kde_bandwidth
    self.density_threshold_ratio = density_threshold_ratio
    self.margin_width = margin_width
    super().__init__()

  def _filter_candidates_by_percentile(
      self,
      confidences: np.ndarray,
  ) -> np.ndarray:
    """Selects indices of elements in the array that are below a certain percentile.

    Args:
        confidences: A 1D numpy array representing confidence scores.

    Returns:
        A numpy array of indices of the elements below the specified percentile.
    """
    if confidences.ndim != 1:
      raise ValueError("Input array must be one-dimensional.")
    self.marginal_candidates_threshold = np.percentile(
        confidences, self.marginal_candidates_threshold * PERCENTS
    )

    return np.where(confidences <= self.marginal_candidates_threshold)[0]

  def _filter_candidates_by_gradient_tilt(
      self,
      confidences: np.ndarray,
  ) -> np.ndarray:
    """Selects indices of elements in the array based on the gradient tilt of the histogram.

    Args:
        confidences: A 1D numpy array representing confidence scores.

    Returns:
        A numpy array of indices of the elements below the identified threshold.
    """

    counts, bin_edges = np.histogram(confidences, bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Smooth histogram with Gaussian filter
    smooth_counts = ndimage.gaussian_filter1d(
        counts, sigma=self.sigma_gaussian_filter_score_hist_smoothing
    )

    # Compute derivative (numerical gradient)
    slope = np.gradient(smooth_counts, bin_centers)

    # Find the max absolute derivative (steepest slope)
    max_slope_idx = np.argmax(np.abs(slope))
    steepest_bin_value = bin_centers[max_slope_idx]

    return np.where(confidences <= steepest_bin_value)[0]

  def _filter_candidates_by_gkde(
      self,
      confidences: np.ndarray,
  ) -> np.ndarray:
    """Selects indices of marginal embeddings based on the Gaussian Kernel Density Estimation.

    Args:
        confidences: A 1D numpy array representing confidence scores.

    Returns:
        A numpy array of indices of the elements within the marginal area.
    """

    # KDE for density estimation of similarity scores
    kde = neighbors.KernelDensity(
        kernel="gaussian", bandwidth=self.kde_bandwidth
    ).fit(X=confidences[:, np.newaxis])
    sim_space = np.linspace(
        confidences.min() - 0.05, confidences.max() + 0.05, 1000
    )
    log_density = kde.score_samples(sim_space[:, np.newaxis])
    density = np.exp(log_density)

    # Identify marginal area using density drop
    peak_density_idx = np.argmax(density)
    peak_density = density[peak_density_idx]

    # Set thresholds where density drops to specified ratio of the peak density
    density_threshold = peak_density * self.density_threshold_ratio
    indices_above_threshold = np.where(density >= density_threshold)[0]

    left_idx = indices_above_threshold[0]
    right_idx = indices_above_threshold[-1]

    left_threshold = sim_space[left_idx]
    right_threshold = sim_space[right_idx]

    # Marginal area: slightly expand the area near thresholds
    marginal_lower = left_threshold - self.margin_width
    marginal_upper = right_threshold + self.margin_width

    # Find indexes within the marginal area
    marginal_indexes = np.where(
        (confidences >= marginal_lower) & (confidences <= marginal_upper)
    )[0]

    return marginal_indexes

  @overrides
  def sample(
      self,
      embeddings: np.ndarray,
      confidences: np.ndarray,
      num_samples: int,
  ) -> np.ndarray:
    """Applies the algorithm to select few-shot samples based on confidence and diversity.

    Args:
        embeddings: A numpy array of image embeddings.
        confidences: A numpy array of confidence scores, representing the
          probability of each image belonging to the target class.
        num_samples: The number of samples to select.

    Returns:
        A numpy array of the selected image embeddings indices (few shot
        samples).

    Raises:
        ValueError: If the embeddings or confidences arrays have incorrect
        dimensions.
    """
    if embeddings.ndim != 2:
      raise ValueError("Embeddings array must be two-dimensional.")
    if confidences.ndim != 1:
      raise ValueError("Confidences array must be one-dimensional.")

    candidate_indices = self.filtering_method(confidences)

    filtering_marginal_candidates_pred_embeddings = np.array(
        embeddings[candidate_indices]
    )
    filtering_marginal_candidates_pred_scores = np.array(
        confidences[candidate_indices]
    )

    # Select diversified candidates from the remaining samples.
    _, selected_original_indices = select_diversified_candidates(
        filtering_marginal_candidates_pred_embeddings,
        filtering_marginal_candidates_pred_scores,
        num_samples=num_samples,
        candidate_frac=self.candidate_frac,
        max_num_samples_to_process=self.max_num_samples_to_process,
    )

    # Map back to original indices
    self._selected_indices = candidate_indices[selected_original_indices]

    return self._selected_indices


class CandidateThresholdingMethod(enum.Enum):
  """Enum for candidate thresholding methods."""

  KDE_FROM_PEAK = enum.auto()
  KDE_PURE = enum.auto()


class PCAKDEClusteringSampler(BaseSampler):
  """Selects diverse samples using PCA, KDE, and K-Means.

  This method augments embeddings with confidence scores, reduces their
  dimensionality with PCA, identifies low-density ("marginal") candidates
  using KDE, and finally clusters these candidates to select a diverse and
  representative final set.
  """

  def __init__(
      self,
      candidate_frac: float = 0.7,
      max_num_samples_to_process: int = 500,
      pca_components: int = 6,
      candidates_thresholding_method: CandidateThresholdingMethod = CandidateThresholdingMethod.KDE_FROM_PEAK,
      kde_bandwidth: float = 0.1,
      density_ratio_from_peak: float = 0.6,
      density_threshold_percentile: float = 10,
      bin_size_thresholding_filtering: float = 20.0,
      threshold_automation: bool = True,
  ):
    """Initializes the sampler.

    Args:
        candidate_frac: Fraction of marginal candidates to use for the final
          diversified selection.
        max_num_samples_to_process: Max candidates to process with K-Means.
        pca_components: Number of principal components for PCA reduction.
        candidates_thresholding_method: Method for marginal region detection,
          either 'kde_from_peak' or 'kde_pure'.
        kde_bandwidth: Bandwidth for the Kernel Density Estimation.
        density_ratio_from_peak: Density ratio from the peak used as a threshold
          when method is 'kde_pure'.
        density_threshold_percentile: Density percentile used as a threshold
          when method is 'kde_from_peak'.
        bin_size_thresholding_filtering: Number of bins for iterative threshold
          adjustment.
        threshold_automation: Whether to automatically adjust the density
          threshold to find enough candidates.
    """
    if not 0 < candidate_frac <= 1:
      raise ValueError("candidate_frac must be in the interval (0, 1].")

    # Store parameters
    self.candidate_frac = candidate_frac
    self.max_num_samples_to_process = max_num_samples_to_process
    self.pca_n_components = pca_components
    self.candidates_thresholding_method = candidates_thresholding_method
    self.kde_bandwidth = kde_bandwidth
    self.density_ratio_from_peak = density_ratio_from_peak
    self.density_threshold_percentile = density_threshold_percentile
    self.bin_size_thresholding_filtering = bin_size_thresholding_filtering
    self.threshold_automation = threshold_automation
    super().__init__()

  def _find_marginal_candidates(
      self, pca_embeddings: np.ndarray, num_samples: int
  ) -> np.ndarray:
    """Identifies candidates in low-density regions of the embedding space."""
    kde = neighbors.KernelDensity(
        kernel="gaussian", bandwidth=self.kde_bandwidth
    ).fit(pca_embeddings)
    log_density = kde.score_samples(pca_embeddings)

    # Use local variables for thresholding to avoid modifying instance state.
    density_percentile = self.density_threshold_percentile
    density_ratio = self.density_ratio_from_peak

    max_iterations = 10
    candidate_indices = np.array([])
    for _ in range(max_iterations):
      if (
          self.candidates_thresholding_method
          == CandidateThresholdingMethod.KDE_FROM_PEAK
      ):
        threshold = np.percentile(log_density, density_percentile)
      else:  # KDE_PURE
        peak_log_density = np.max(log_density)
        threshold = peak_log_density + np.log(density_ratio)

      candidate_indices = np.where(log_density < threshold)[0]

      # Exit if enough samples are found or if automation is disabled.
      if candidate_indices.size >= num_samples or not self.threshold_automation:
        return candidate_indices

      # If automating, adjust local threshold parameters for the next iteration.
      if self.candidates_thresholding_method == "kde_from_peak":
        density_percentile += PERCENTS / self.bin_size_thresholding_filtering
        if density_percentile >= 100:
          break
      else:
        # kde_pure
        density_ratio += 1.0 / self.bin_size_thresholding_filtering
        if density_ratio >= 1.0:
          break

    logging.warning(
        "Could not find enough marginal candidates after %d attempts. Found %d,"
        " needed %d. Consider adjusting KDE or thresholding parameters.",
        max_iterations,
        candidate_indices.size,
        num_samples,
    )
    return candidate_indices

  @overrides
  def sample(
      self,
      embeddings: np.ndarray,
      confidences: np.ndarray,
      num_samples: int,
  ) -> np.ndarray:
    """Selects diverse, low-confidence samples using the PCA-KDE pipeline.

    Args:
        embeddings: A 2D numpy array of item embeddings.
        confidences: A 1D numpy array of confidence scores for each item.
        num_samples: The target number of samples to select.

    Returns:
        A 1D numpy array of indices corresponding to the selected samples.
    """
    if embeddings.ndim != 2:
      raise ValueError("Embeddings array must be 2D.")
    if confidences.ndim != 1 or len(embeddings) != len(confidences):
      raise ValueError(
          "Confidences must be a 1D array of the same length as embeddings."
      )

    # Augment embeddings with confidences and apply PCA.
    augmented_embeddings = np.hstack([
        embeddings,
        confidences.reshape(-1, 1),
    ])
    pca = decomposition.PCA(n_components=self.pca_n_components)
    pca_embeddings = pca.fit_transform(augmented_embeddings)

    # Find marginal candidates using KDE on the PCA-reduced data.
    marginal_indices = self._find_marginal_candidates(
        pca_embeddings, num_samples
    )
    if marginal_indices.size < num_samples:
      raise ValueError(
          f"Found only {marginal_indices.size} marginal candidates, but "
          f"require {num_samples} to proceed. Adjust KDE/thresholding params."
      )

    # Select a diversified subset from the marginal candidates.
    candidate_embeddings = embeddings[marginal_indices]
    candidate_scores = confidences[marginal_indices]

    # Determine the fraction of candidates to pass to the diversifier locally.
    final_candidate_frac = self.candidate_frac
    if (final_candidate_frac * candidate_scores.size) < num_samples:
      final_candidate_frac = min(
          1.0,
          num_samples / candidate_scores.size + EPSILON_FOR_CANDIDATE_FRACTION,
      )

    _, diversified_indices = select_diversified_candidates(
        embeddings=candidate_embeddings,
        similarity_scores=candidate_scores,
        num_samples=num_samples,
        candidate_frac=final_candidate_frac,
        max_num_samples_to_process=self.max_num_samples_to_process,
    )

    # Map indices back to the original dataset and return.
    self.selected_indices_ = marginal_indices[diversified_indices]
    return self.selected_indices_
