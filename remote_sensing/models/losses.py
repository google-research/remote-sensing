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


class BinarySegmentationDiceLoss(nn.Module):
  """Dice loss for binary segmentation.

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
      logits: The raw logits from the model. Expected shape is (B, H, W) or (B,
        1, H, W). If shape is (B, 1, H, W), the class dimension is squeezed.
      targets: The ground truth segmentation masks. Expected shape is (B, H, W).
      mask: An optional mask tensor to apply to the loss computation. If
        provided, loss is only computed for masked pixels. Expected shape is (B,
        H, W).

    Returns:
      The mean Dice loss over the batch.
    """
    if logits.dim() not in (3, 4):
      raise ValueError(
          'Logits must have shape (B, H, W) or (B, 1, H, W), got'
          f' {logits.shape}.'
      )
    if targets.dim() != 3:
      raise ValueError(
          f'Targets must have shape (B, H, W), got {targets.shape}.'
      )
    if logits.dim() == 4:
      if logits.shape[1] == 1:
        logits = logits.squeeze(1)
      else:
        raise ValueError(
            f'Logits must have 1 class, but has {logits.shape[1]} classes.'
        )
    targets = targets.float()

    probs = torch.sigmoid(logits)

    dims = (1, 2) if self.per_image_loss else (0, 1, 2)

    if mask is None:
      # Sum over spatial dimensions H and W to compute mask overlap.
      intersection = (probs * targets).sum(dim=dims)
      p_sum = probs.sum(dim=dims)
      t_sum = targets.sum(dim=dims)
    else:
      if mask.dim() != 3:
        raise ValueError(f'Mask must have shape (B, H, W), got {mask.shape}.')
      mask = mask.float()
      intersection = (probs * targets * mask).sum(dim=dims)
      p_sum = (probs * mask).sum(dim=dims)
      t_sum = (targets * mask).sum(dim=dims)
    dice = (2.0 * intersection + self.epsilon) / (p_sum + t_sum + self.epsilon)

    loss = 1.0 - dice
    return loss.mean() if self.per_image_loss else loss


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
    self.losses = nn.ModuleList(losses)
    self.weights = weights

  def forward(
      self, logits: torch.Tensor, targets: torch.Tensor
  ) -> torch.Tensor:
    """Computes the weighted sum of losses.

    Args:
      logits: The raw logits from the model.
      targets: The ground truth targets.

    Returns:
      The combined loss over the batch.
    """
    total_loss = torch.tensor(0.0, device=logits.device)
    for weight, loss_fn in zip(self.weights, self.losses):
      total_loss += weight * loss_fn(logits, targets)
    return total_loss
