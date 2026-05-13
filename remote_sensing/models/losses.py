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

from typing import Optional
import torch
from torch import nn
from torchvision import ops

sigmoid_focal_loss = ops.sigmoid_focal_loss


class SegmentationDiceLoss(nn.Module):
  """Dice loss for segmentation.

  Supports binary and multi-class segmentation.
  Dice loss optimizes for region overlap between the predicted segmentation
  mask and the ground truth mask by summing over the spatial dimensions.
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
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Computes the Dice loss.

    Args:
      logits: The raw logits from the model. Expected shape is (B, C, H, W).
        If C=1, binary segmentation is assumed, if C>1, multi-class.
      targets: The ground truth segmentation masks, one-hot encoded.
        Expected shape is (B, C, H, W).
      mask: An optional mask tensor to apply to the loss computation. If
        provided, loss is only computed for masked pixels. Expected shape is (B,
        1, H, W).

    Returns:
      The mean Dice loss.
    """
    if logits.dim() != 4:
      raise ValueError(
          f'Logits must have shape (B, C, H, W), got {logits.shape}.'
      )
    if targets.dim() != 4:
      raise ValueError(
          f'Targets must have shape (B, C, H, W), got {targets.shape}.'
      )
    if logits.shape != targets.shape:
      raise ValueError(
          f'Logits shape {logits.shape} and targets shape {targets.shape} must'
          ' be identical.'
      )

    num_classes = logits.shape[1]

    if num_classes == 1:
      probs = torch.sigmoid(logits)
    else:  # num_classes > 1
      probs = torch.softmax(logits, dim=1)

    targets = targets.float()

    dims = (2, 3) if self.per_image_loss else (0, 2, 3)

    if mask is None:
      intersection = (probs * targets).sum(dim=dims)
      p_sum = probs.sum(dim=dims)
      t_sum = targets.sum(dim=dims)
    else:
      if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f'Mask must have shape (B, 1, H, W), got {mask.shape}.'
        )

      mask = mask.float()
      intersection = (probs * targets * mask).sum(dim=dims)
      p_sum = (probs * mask).sum(dim=dims)
      t_sum = (targets * mask).sum(dim=dims)

    dice = (2.0 * intersection + self.epsilon) / (p_sum + t_sum + self.epsilon)

    loss = 1.0 - dice
    return loss.mean()


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
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Computes the focal loss.

    Args:
      logits: The raw logits from the model. Expected shape is (B, C, H, W).
      targets: The ground truth segmentation masks, one-hot encoded.
        Expected shape is (B, C, H, W).
      mask: An optional mask tensor. Expected shape is (B, 1, H, W).

    Returns:
      The Focal loss over the batch.
    """
    if logits.dim() != 4:
      raise ValueError(
          f'Logits must have shape (B, C, H, W), got {logits.shape}.'
      )
    if targets.dim() != 4:
      raise ValueError(
          f'Targets must have shape (B, C, H, W), got {targets.shape}.'
      )
    if logits.shape != targets.shape:
      raise ValueError(
          f'Logits shape {logits.shape} and targets shape {targets.shape} must'
          ' be identical.'
      )

    num_classes = logits.shape[1]
    targets_onehot = targets.float()

    focal_loss = ops.sigmoid_focal_loss(
        logits,
        targets_onehot,
        alpha=self.alpha,
        gamma=self.gamma,
        reduction='none',
    )

    if mask is not None:
      if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(
            f'Mask must have shape (B, 1, H, W), got {mask.shape}.'
        )
      mask_expanded = mask.float()
      focal_loss = focal_loss * mask_expanded
      valid_elements = mask_expanded.sum() * num_classes
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
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Flattens predictions in the batch dimension and filters by mask.

  Args:
    probs: Probability tensor of shape (B, C, H, W).
    labels: Class label tensor of shape (B, H, W).
    mask: Optional mask tensor of shape (B, 1, H, W) or (B, H, W). If provided,
      only elements where mask > 0 are kept.

  Returns:
    A tuple of flattened tensors (probs, labels), where probs has shape
    (N, C) and labels has shape (N,), and N is the number of valid elements.
  """
  if probs.dim() != 4:
    raise ValueError(f'Probs must have shape (B, C, H, W), got {probs.shape}.')
  if labels.dim() != 3:
    raise ValueError(f'Labels must have shape (B, H, W), got {labels.shape}.')

  _, num_classes, _, _ = probs.shape
  probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)  # (B*H*W, C)
  labels = labels.reshape(-1)  # (B*H*W)

  if mask is None:
    return probs, labels
  if mask.dim() == 4:
    if mask.shape[1] != 1:
      raise ValueError(
          f'Mask must have shape (B, 1, H, W) or (B, H, W), got {mask.shape}.'
      )
  elif mask.dim() != 3:
    raise ValueError(
        f'Mask must have shape (B, 1, H, W) or (B, H, W), got {mask.shape}.'
    )

  mask = mask.float().reshape(-1)
  valid = mask > 0
  probs = probs[valid]
  labels = labels[valid]
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
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Computes the Lovasz-Softmax loss.

    Args:
      logits: The raw logits from the model. Expected shape is (B, C, H, W).
      targets: The ground truth segmentation masks, one-hot encoded. Expected
        shape is (B, C, H, W).
      mask: An optional mask tensor. Expected shape is (B, 1, H, W).

    Returns:
      The Lovasz-Softmax loss over the batch.
    """
    if logits.dim() != 4:
      raise ValueError(
          f'Logits must have shape (B, C, H, W), got {logits.shape}.'
      )
    if targets.dim() != 4:
      raise ValueError(
          f'Targets must have shape (B, C, H, W), got {targets.shape}.'
      )
    if logits.shape != targets.shape:
      raise ValueError(
          f'Logits shape {logits.shape} and targets shape {targets.shape} must'
          ' be identical.'
      )
    if logits.shape[1] == 1:
      raise ValueError('Lovasz-Softmax loss requires C>1.')

    labels = targets.argmax(dim=1).long()
    probs = torch.softmax(logits, dim=1)
    probs_flat, labels_flat = _flatten_probs(probs, labels, mask)
    return _lovasz_softmax_flat(
        probs_flat, labels_flat, classes=self.classes
    )


class CombinedLoss(nn.Module):
  """Computes a weighted sum of multiple loss functions.

  This is useful for combining different loss functions, e.g., Focal loss and
  Dice loss for binary segmentation, with
  potentially varying weights across training epochs.
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
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Computes the weighted sum of losses.

    Args:
      logits: The raw logits from the model.
      targets: The ground truth targets.
      mask: An optional mask tensor passed to each loss function.

    Returns:
      The combined loss over the batch.
    """
    total_loss = torch.tensor(0.0, device=logits.device)

    for weight, loss_fn in zip(self.weights, self.losses):
      total_loss += weight * loss_fn(logits, targets, mask)
    return total_loss

