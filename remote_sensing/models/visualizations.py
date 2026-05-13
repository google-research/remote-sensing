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

"""Utilities for visualizing different outputs of models."""

import dataclasses
import math
import torch


class Denormalize(torch.nn.Module):
  """The inverse of the Normalize transform.

  Used for visualizing normalized images.

  normalized = (original - MEAN) / STD
  original = normalized * STD + MEAN
  """

  def __init__(self, mean: torch.Tensor, std: torch.Tensor):
    super().__init__()
    self.mean = mean
    self.std = std

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = self.mean.to(x.device)[None, :, None, None]
    std = self.std.to(x.device)[None, :, None, None]
    return (x * std) + mean


def auto_label_palette(num_classes: int) -> torch.Tensor:
  """Automatically generates a color palette for the given number of classes.

  Args:
    num_classes: The number of classes.

  Returns:
    A torch tensor of shape [num_classes, 3] with the color palette.
    The palette should have a good contrast between each two colors.
  """
  # All shades or R/G/B.
  v = torch.linspace(0, 1.0, math.ceil(num_classes ** (1 / 3.0)))
  # All combinations of R/G/B, with at least num_classes combinations.
  # Notice that ceil(c ** 1/3.)**3 >= c.
  color_set = torch.stack(torch.meshgrid(v, v, v, indexing='ij'), dim=3)
  color_set = color_set.reshape(-1, 3)
  # Stores the minimum distance from each color to the set of chosen colors.
  distances = torch.full((len(color_set),), torch.inf)
  label_palette = []
  # Add the most distant color from the set, until we have enough colors.
  while len(label_palette) < num_classes:
    new = color_set[torch.argmax(distances)]
    label_palette.append(new)
    # Update the distances from each color to the set of chosen colors.
    distances = torch.min(distances, torch.linalg.norm(color_set - new, dim=1))
  return torch.stack(label_palette, dim=0).to(torch.float32)


@dataclasses.dataclass
class SegmentationVisualizationConfig:
  """Config for the visualization.

  The defaults values should work for most use cases, by using `auto`.
  """

  @classmethod
  def auto(cls, num_classes: int) -> 'SegmentationVisualizationConfig':
    return cls(label_palette=auto_label_palette(num_classes))

  # A color palette for the labels, of shape [num_classes, 3], in range [0, 1].
  # use 'auto_label_palette(num_classes)' to generate a default palette.
  label_palette: torch.Tensor

  # A color palette for the correctness, of shape [2, 3], in range [0, 1].
  # The first color is for incorrect pixels, the second for correct pixels.
  correct_palette: torch.Tensor = torch.tensor([
      [1.0, 0.6, 0.6],  # Red = Wrong
      [0.6, 1.0, 0.6],  # Green = Correct
  ])

  # The ratio of the blend between the original images and the mappings.
  blend_ratio: float = 0.5


def visualize_segmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor | None = None,
    config: SegmentationVisualizationConfig | None = None,
) -> torch.Tensor:
  """Visualizes images with segmentation labels and predictions.

  For each image in the batch, outputs a row of four tiles:
  [image, label, prediction, correctness].
  The label and the predition are visualized using the `label_palette`, where
  each pixel's color is a weighted sum of the palette colors, weighted by the
  classes probabilities. The correctness is the agreement between the labels and
  the predictions, and is visualized using the `correct_palette`.
  The label, prediction, and correctness maps are then blended with the
  original images using the `blend_ratio`, and the `weights` if provided.

  Args:
    images: A tensor of shape [batch, 3, height, width] in range [0, 1]. Use
      `Denormalize` to denormalize the images if needed.
    labels: A tensor of shape [batch, num_classes, height, width]. The tensor is
      a probability distribution over the classes for each pixel.
    predictions: A tensor of shape [batch, num_classes, height, width]. The
      tensor is a probability distribution over the classes for each pixel.
    weights: An optional tensor of shape [batch, 1, height, width]. If provided,
      the weights represent the certainty of the labels, in the range [0, 1]. If
      not provided, the weight is assumed to be 1.
    config: The configuration for the visualization, or None to use the default
      values.

  Returns:
    An image, as a tensor of shape [3, batch * height, 4 * width], with the
    following visualizations: [
        Image_1, Label_1, Prediction_1, Correctness_1,
        ...
        Image_N, Label_N, Prediction_N, Correctness_N,
    ]
  """
  if config is None:
    config = SegmentationVisualizationConfig.auto(num_classes=labels.shape[1])

  # Visualze the labels and the predictions, based on the label palette.
  labels_map = torch.einsum('bchw,cd->bdhw', labels, config.label_palette)
  preds_map = torch.einsum('bchw,cd->bdhw', predictions, config.label_palette)

  # Visualize the correctness based on the correct_palette.
  # For hard-labeled pixels (e.g. a pixel is assigned to a single class), this
  # ends up being the predicted probability of the correct class.
  # For soft-labeled pixels (e.g. a pixel is assigned to multiple classes),
  # this is the intersection over union of all the class probabilities.
  correct = torch.sum(
      torch.min(labels, predictions), dim=1, keepdim=True
  ) / torch.sum(torch.max(labels, predictions), dim=1, keepdim=True)
  correct = torch.concatenate([1 - correct, correct], dim=1)
  correct_map = torch.einsum('bchw,cd->bdhw', correct, config.correct_palette)

  blend = config.blend_ratio
  if weights is not None:
    blend *= weights

  labels_map = labels_map * blend + images * (1 - blend)
  preds_map = preds_map * blend + images * (1 - blend)
  correct_map = correct_map * blend + images * (1 - blend)

  # Images, labels, predictions, correctness are of shape [B, C, H, W].
  # Concatenate them to [B, C, H, 4 * W].
  rows = torch.concatenate([images, labels_map, preds_map, correct_map], dim=3)
  rows = torch.nan_to_num(rows, nan=0, posinf=0, neginf=0)
  rows = torch.clamp(rows, 0, 1)
  # Permute them to [C, B, H, 4 * W], and reshape to [C, B * H, 4 * W].
  return rows.permute(1, 0, 2, 3).reshape(3, -1, rows.shape[-1])
