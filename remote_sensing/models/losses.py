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

"""Loss functions for imbalanced semantic segmentation."""

from collections.abc import Callable
import torch
from torch import nn
from torchvision import ops

sigmoid_focal_loss = ops.sigmoid_focal_loss


def _validate_shapes(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
):
  """Validates the shapes of the logits, targets, and weight tensors.

  The logits must be a 3+D tensor of shape ([B...], C, H, W).
  The targets must be a 3+D tensor of shape ([B...], C, H, W).
  The weight can be None or a 3+D tensor of shape ([B...], 1, H, W).

  The values of [B...] (optional extra dimensions), C (number of classes), H
  (height) and W (width) must be consistent across the tensors.

  Args:
    logits: The raw logits from the model.
    target: The ground truth targets.
    weight: An optional weight tensor.

  Raises:
    ValueError: If the shapes of the logits, targets, or weight tensors are
      invalid.
  """
  if logits.dim() < 3:
    raise ValueError(
        f'Logits shape must be ([B...], C, H, W), got {logits.shape}.'
    )
  if target.dim() < 3:
    raise ValueError(
        f'Target shape must be ([B...], C, H, W), got {target.shape}.'
    )
  if logits.shape != target.shape:
    raise ValueError(
        f'Logits shape {logits.shape} and targets shape {target.shape} must be '
        'identical.'
    )
  if weight is not None:
    if weight.dim() < 3:
      raise ValueError(
          f'Weight must have shape ([B...], 1, H, W), got {weight.shape}.'
      )
    if weight.shape[-3] != 1:
      raise ValueError(
          f'Weight must have shape ([B...], 1, H, W), got {weight.shape}.'
      )
    b = logits.shape[:-3]
    h = logits.shape[-2]
    w = logits.shape[-1]
    if weight.shape != (*b, 1, h, w):
      raise ValueError(
          f'Weight shape {weight.shape} must match logits shape'
          f' {logits.shape} in batch, height, and width dimensions.'
      )


def _aggregation_dims(
    rank: int,
    per_image: bool = False,
) -> tuple[int, ...]:
  dims = (rank - 2, rank - 1)
  if not per_image:
    dims = tuple(range(rank - 3)) + dims
  return dims


class SegmentationDiceLoss(nn.Module):
  """Dice loss for segmentation.

  Supports binary and multi-class segmentation.
  Dice loss optimizes for region overlap between the predicted segmentation
  weight and the ground truth weight by summing over the spatial dimensions.
  It is effective when overlap-based performance is important.
  """

  def __init__(self, epsilon: float = 1e-6, per_image_loss: bool = False):
    """Initializes the Dice loss.

    Args:
      epsilon: A small epsilon value to add to the numerator and denominator to
        avoid division by zero.
      per_image_loss: If True, calculates Dice loss per image and averages over
        the batch. If False, calculates Dice loss over the entire batch.
    """
    super().__init__()
    self.epsilon = epsilon
    self.per_image_loss = per_image_loss

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes the Dice loss.

    if `per_image_loss` is True, the loss is calculated per image and then
    averaged over the batch. Otherwise, the loss is calculated over the entire
    batch.

    Args:
      logits: The raw logits from the model. Expected shape is ([B...], C, H,
        W). If C=1, binary segmentation is assumed, if C>1, multi-class.
      targets: The ground truth segmentation masks, one-hot encoded. Expected
        shape is ([B...], C, H, W).
      weight: An optional weight tensor to apply to the loss computation. If
        provided, each pixel has its own weight. Expected shape is ([B...], 1,
        H, W).

    Returns:
      The mean Dice loss (a single scalar value).
    """
    _validate_shapes(logits, targets, weight)

    num_classes = logits.shape[-3]

    if num_classes == 1:
      probs = torch.sigmoid(logits)
    else:  # num_classes > 1
      probs = torch.softmax(logits, dim=1)

    targets = targets.float()

    dims = _aggregation_dims(logits.dim(), self.per_image_loss)

    if weight is None:
      intersection = (probs * targets).sum(dim=dims)
      p_sum = probs.sum(dim=dims)
      t_sum = targets.sum(dim=dims)
    else:
      weight = weight.float()
      intersection = (probs * targets * weight).sum(dim=dims)
      p_sum = (probs * weight).sum(dim=dims)
      t_sum = (targets * weight).sum(dim=dims)

    dice = (2.0 * intersection + self.epsilon) / (p_sum + t_sum + self.epsilon)

    loss = 1.0 - dice
    return loss.mean()


class SegmentationJaccardLoss(nn.Module):
  """Jaccard loss for segmentation.

  Supports binary and multi-class segmentation.
  Jaccard loss optimizes directly for Intersection over Union (IoU).
  """

  def __init__(self, epsilon: float = 1e-6, per_image_loss: bool = False):
    """Initializes the Jaccard loss.

    Args:
      epsilon: A small epsilon value to add to the numerator and denominator to
        avoid division by zero.
      per_image_loss: Whether to calculates Jaccard loss per image or per batch.
    """
    super().__init__()
    self.epsilon = epsilon
    self.per_image_loss = per_image_loss

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes the Jaccard loss.

    if `per_image_loss` is True, the loss is calculated per image and then
    averaged over the batch. Otherwise, the loss is calculated over the entire
    batch.

    Args:
      logits: The raw logits from the model. Expected shape is ([B...], C, H,
        W). If C=1, binary segmentation is assumed, if C>1, multi-class.
      targets: The ground truth segmentation masks, one-hot encoded. Expected
        shape is ([B...], C, H, W).
      weight: An optional weight tensor to apply to the loss computation. If
        provided, each pixel has its own weight. Expected shape is ([B...], 1,
        H, W).

    Returns:
      The mean Jaccard loss (a single scalar value).
    """
    _validate_shapes(logits, targets, weight)

    if logits.shape[-3] == 1:
      probs = torch.sigmoid(logits)
    else:  # num_classes > 1
      probs = torch.softmax(logits, dim=1)

    targets = targets.float()
    intersection = probs * targets
    sum_probs_targets = probs + targets
    if weight is not None:
      intersection *= weight
      sum_probs_targets *= weight

    dims = _aggregation_dims(logits.dim(), self.per_image_loss)

    intersection = intersection.sum(dim=dims)
    union = sum_probs_targets.sum(dim=dims) - intersection
    iou = (intersection + self.epsilon) / (union + self.epsilon)
    return 1.0 - iou.mean()


class SegmentationFocalLoss(nn.Module):
  """Focal loss for segmentation.

  Supports binary and multi-class segmentation.

  Paper: https://arxiv.org/abs/1708.02002
  """

  def __init__(
      self,
      alpha: float = 0.25,
      gamma: float = 2.0,
  ):
    """Initializes the Focal loss.

    Args:
      alpha: Alpha parameter for focal loss.
      gamma: Gamma parameter for focal loss.
    """
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes the focal loss.

    Args:
      logits: The raw logits from the model. Expected shape is ([B...], C, H,
        W).
      targets: The ground truth segmentation masks, one-hot encoded. Expected
        shape is ([B...], C, H, W).
      weight: An optional weight tensor. Expected shape is ([B...], 1, H, W).

    Returns:
      The Focal loss over the batch.
    """
    _validate_shapes(logits, targets, weight)

    num_classes = logits.shape[-3]
    targets_onehot = targets.float()

    focal_loss = ops.sigmoid_focal_loss(
        logits,
        targets_onehot,
        alpha=self.alpha,
        gamma=self.gamma,
        reduction='none',
    )

    if weight is not None:
      weight_expanded = weight.float()
      focal_loss = focal_loss * weight_expanded
      valid_elements = weight_expanded.sum() * num_classes
      focal_loss = focal_loss.sum() / valid_elements.clamp(min=1.0)
    else:
      focal_loss = focal_loss.mean()

    return focal_loss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
  """Computes gradient of Lovasz extension with respect to logits.

  Args:
    gt_sorted: A 1D tensor of ground truth labels (0.0 or 1.0) for a single
      class, sorted in descending order by the magnitude of errors (|gt -
      pred|). Expected shape is (N,).

  Returns:
    The gradient of the Lovasz extension.
  """
  num_elements = gt_sorted.numel()
  gt_sum = gt_sorted.sum()
  intersection = gt_sum - gt_sorted.float().cumsum(0)
  union = gt_sum + (1.0 - gt_sorted).float().cumsum(0)
  jaccard = 1.0 - intersection / union.clamp_min(1e-6)

  if num_elements > 1:
    jaccard[1:num_elements] = jaccard[1:num_elements] - jaccard[0:-1]
  return jaccard


def _flatten_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Flattens predictions in the batch dimension and filters by weight.

  Args:
    probs: Probability tensor of shape ([B...], C, H, W).
    labels: Class label tensor of shape ([B...], B, H, W).
    mask: Optional boolean mask tensor of shape ([B...], 1, H, W). If provided,
      only positive elements are kept.

  Returns:
    A tuple of flattened tensors (probs, labels), where probs has shape
    (N, C) and labels has shape (N,), and N is the number of valid elements.
  """
  _, num_classes, _, _ = probs.shape
  probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)  # (B*H*W, C)
  labels = labels.reshape(-1)  # (B*H*W)

  if mask is None:
    return probs, labels
  mask = mask.reshape(-1)
  probs = probs[mask]
  labels = labels[mask]
  return probs, labels


def _lovasz_softmax_flat(
    probs: torch.Tensor, labels: torch.Tensor, classes: str = 'present'
) -> torch.Tensor:
  """Computes Lovasz-Softmax loss from flattened probabilities and labels.

  Args:
    probs: Flattened probability tensor of shape (N, C), where N is the number
      of elements and C is the number of classes.
    labels: Flattened class label tensor of shape (N,).
    classes: 'present' or 'all'. If 'present', only classes present in labels
      are considered for loss computation. If 'all', all classes are considered.

  Returns:
    The Lovasz-Softmax loss.
  """
  if probs.numel() == 0:
    return probs.sum() * 0.0

  num_classes = probs.shape[1]
  losses = []
  for class_index in range(num_classes):
    class_targets = torch.as_tensor(labels == class_index, dtype=torch.float32)
    if classes == 'present' and class_targets.sum() == 0:
      continue

    class_pred = probs[:, class_index]
    errors = (class_targets - class_pred).abs()
    errors_sorted, perm = torch.sort(errors, 0, descending=True)
    perm = perm.data
    class_targets_sorted = class_targets[perm]
    losses.append(torch.dot(errors_sorted, _lovasz_grad(class_targets_sorted)))

  if not losses:
    return probs.sum() * 0.0
  return torch.stack(losses).mean()


class SegmentationLovaszSoftmaxLoss(nn.Module):
  """Lovasz-Softmax loss for segmentation.

  Supports multi-class segmentation.
  It is a loss function for semantic segmentation that directly optimizes
  the Jaccard index (IoU) for multi-class problems.
  It is particularly effective for tasks with imbalanced classes.

  Paper: https://arxiv.org/abs/1705.08790
  """

  def __init__(self, classes: str = 'present'):
    """Initializes the Lovasz-Softmax loss.

    Args:
      classes: 'present' or 'all'. If 'present', only classes present in the
        ground truth are considered for loss computation. If 'all', all classes
        are considered.
    """
    super().__init__()
    if classes not in ['present', 'all']:
      raise ValueError(f"classes must be 'present' or 'all', got {classes}.")
    self.classes = classes

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes the Lovasz-Softmax loss.

    Args:
      logits: The raw logits from the model. Expected shape is ([B...], C, H,
        W).
      targets: The ground truth segmentation masks, one-hot encoded. Expected
        shape is ([B...], C, H, W).
      weight: An optional weight tensor. Expected shape is ([B...], B, 1, H, W).
        note that unlike other losses, the weight is only used as a binary mask,
        so weight > 0 is treated as 1, and weight = 0 is treated as 0.

    Returns:
      The Lovasz-Softmax loss over the batch.
    """
    _validate_shapes(logits, targets, weight)

    labels = targets.argmax(dim=1).long()
    probs = torch.softmax(logits, dim=1)
    mask = weight > 0 if weight is not None else None
    probs_flat, labels_flat = _flatten_probs(probs, labels, mask)
    return _lovasz_softmax_flat(probs_flat, labels_flat, classes=self.classes)


class SegmentationCrossEntropyLoss(nn.Module):
  """A Cross-Entropy loss with the commmon API of segmentation losses."""

  def __init__(self, epsilon: float = 1e-6):
    super().__init__()
    self.xent = torch.nn.CrossEntropyLoss(reduction='none')
    self.epsilon = epsilon

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    _validate_shapes(logits, targets, weight)
    per_pixel_xent = self.xent(logits, targets)
    if weight is None:
      return per_pixel_xent.mean()
    return (per_pixel_xent * weight).sum() / (weight.sum() + self.epsilon)


class CombinedLoss(nn.Module):
  """Computes a weighted sum of multiple loss functions.

  This is useful for combining different loss functions, e.g., Focal loss and
  Dice loss for binary segmentation.
  """

  def __init__(
      self,
      losses: list[nn.Module],
      weights: list[float],
  ):
    """Initializes the combined loss.

    Args:
      losses: A list of loss modules.
      weights: A list of weights corresponding to each loss.
    """
    super().__init__()
    if not losses:
      raise ValueError('Losses cannot be empty.')
    if len(losses) != len(weights):
      raise ValueError('Losses and weights must have the same length.')
    # Use a standard list to allow both functions and modules.
    self.losses = losses
    self.weights = weights

  def forward(
      self,
      logits: torch.Tensor,
      targets: torch.Tensor,
      weight: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Computes the weighted sum of losses.

    Args:
      logits: The raw logits from the model.
      targets: The ground truth targets.
      weight: An optional weight tensor passed to each loss function.

    Returns:
      The combined loss over the batch.
    """
    total_loss = torch.tensor(0.0, device=logits.device)

    for loss_weight, loss_fn in zip(self.weights, self.losses):
      total_loss += loss_weight * loss_fn(logits, targets, weight)
    return total_loss


class ProgressiveCombinedLoss(CombinedLoss):
  """A combined loss where the weights progress over time.

  The weights are determined by a weights_provider function that takes the
  current iteration and total iterations as input and returns a list of weights.

  The user should call step() in order to update the weights.
  Notice that 'iteration' can be a single batch, or an entire epoch, depending
  on the user's implementation.

  Attributes:
    losses: The list of loss modules.
    weights: The list of weights corresponding to each loss.
    weights_provider: The function that provides the weights. Takes current
      iteration and total iterations as input and returns a list of weights.
    total_iterations: The total number of training iterations.
    current_iteration: The current training iteration.
  """

  def __init__(
      self,
      losses: list[nn.Module],
      weights_provider: Callable[[int, int], list[float]],
      total_iterations: int,
  ):
    super().__init__(losses, weights=weights_provider(0, total_iterations))
    self.weights_provider = weights_provider
    self.total_iterations = total_iterations
    self.current_iteration = 0

  def step(self) -> None:
    """Updates the weights to the next iteration."""
    self.current_iteration += 1
    self.weights = self.weights_provider(
        self.current_iteration, self.total_iterations
    )
