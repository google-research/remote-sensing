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

