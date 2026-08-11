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

"""Utility functions for augmentations.

Currently only contains segmentation-specific augmentations.

Usage example:

NUM_CLASSES = 17

transforms = torchvision.transforms.Compose([
    # One-hot encode the label and generate weight maps.
    PrepareSegmentationLabelAndWeight(NUM_CLASSES),

    # Color jitter on the image only.
    ApplyOnField(
        "image",
        torchvision.transforms.v2.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),
    ),

    # Stack dictionary to a single tensor.
    # This is needed to apply augmentations (such as Cropping, Resizing,
    # Flipping, Rotation, CutMix, etc.) to all three tensors in one go.
    StackSegmentationImageLabelWeight(),

    # Random resized crop on the image+label+weight together.
    torchvision.transforms.RandomResizedCrop(
          size=(512, 512),
          antialias=True,
          scale=(0.2, 1.0),
          ratio=(1.0, 1.0),
      ),

    # Random horizontal flip on the image+label+weight together.
    torchvision.transforms.v2.RandomHorizontalFlip(),

    # Random rotation on the image+label+weight together. Padding all with
    # zeros, so the image will have black corners, and the mask will have
    # zeros in the corners.
    # Note that if using this, the loss function must take masks into account.
    torchvision.transforms.v2.RandomRotation([-180, 180]),

    # Random CutMix and MixUp on the image+label+weight together.
    SegmentationCutMixAndMixUp(cutmix_alpha=0.7, mixup_alpha=0.5),

    # Split the dictionary back into three separate tensors.
    SplitSegmentationImageLabelWeight(),

    # Normalize the image only.
    ApplyOnField(
        "image",
        torchvision.transforms.Normalize(
              mean=torch.Tensor([0.485, 0.456, 0.406]),
              std=torch.Tensor([0.229, 0.224, 0.225]),
          ),
    ),

    # Validation check on the shapes and values of the data.
    ValidationCheck(num_channels=3, num_classes=NUM_CLASSES),
])

inputs = {
    "image": ... # shape (B, 3, H, W),
    "label": torch.randint(0, NUM_CLASSES, size=(B, H, W)),
}
outputs = transforms(inputs)
assert outputs["image"].shape == (B, 3, 512, 512)
assert outputs["label"].shape == (B, NUM_CLASSES, 512, 512)
assert outputs["weight"].shape == (B, 1, 512, 512)

Important note:
This augmentation pipeline is designed to generate soft-labels, with per-pixel
weights. If using it as is, it's critical that the weight tensor is used in the
loss function, and that the loss function is compatible with soft-labels.
The `losses.py` library is designed to work with these augmentations.
"""

import random
from typing import Any
import cv2
import torch
from torch import nn
import torchvision
from torchvision.transforms import v2 as torchvision_v2


class PrepareSegmentationLabelAndWeight(nn.Module):
  """Prepares the label and weight tensors for segmentation tasks."""

  def __init__(self, num_classes: int):
    super().__init__()
    self.num_classes = num_classes

  def forward(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One-hot encodes the label, and generates weight maps.

    The input dict `d` is expected to contain:
    - 'image': a float tensor of shape [..., C, H, W]
    - 'label': an int tensor of shape [..., H, W]

    The output dict will contain:
    - 'image': a float tensor of shape [..., C, H, W]
    - 'label': a float tensor of shape [..., num_classes, H, W]
    - 'weight': a float tensor of shape [..., 1, H, W]

    Args:
        d: Input dictionary containing image and label tensors.

    Returns:
        Output dictionary with image, one-hot label, and weight tensors.
    """
    label = d["label"]
    label = nn.functional.one_hot(
        label.long(), num_classes=self.num_classes
    ).float()
    weight = torch.ones(
        label.shape[:-1] + (1,), dtype=torch.float32, device=label.device
    )

    label = torch.movedim(label, -1, -3)
    weight = torch.movedim(weight, -1, -3)

    return {
        "image": d["image"],
        "label": label,
        "weight": weight,
    }


class StackSegmentationImageLabelWeight(nn.Module):
  """Stacks tensors for segmentation tasks.

  By stacking the image, label, and weight tensors along the channel dimension,
  we can apply augmentations to all three tensors in one go.
  """

  def forward(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stacks image, label, and weight tensors along the channel dimension.

    The input dict `d` is expected to contain:
    - 'image': a float tensor of shape [..., C, H, W]
    - 'label': a float tensor of shape [..., D, H, W]
    - 'weight': a float tensor of shape [..., 1, H, W]

    The output tensor will have shape [..., C + D + 1, H, W].

    Args:
        d: Input dictionary containing image, label, and weight tensors.

    Returns:
        A single tensor with image, label, and weight concatenated
        along dim=-3.
    """
    return torch.cat([d["image"], d["label"], d["weight"]], dim=-3)


class SplitSegmentationImageLabelWeight(nn.Module):
  """Reverses the `StackImageLabelWeight` operation."""

  def __init__(self, num_channels: int = 3):
    super().__init__()
    self.num_channels = num_channels

  def forward(self, stacked: torch.Tensor) -> dict[str, torch.Tensor]:
    """Splits a stacked tensor into image, label, and weight tensors.

    The input tensor `stacked` is expected to have shape [..., C + D + 1, H, W].

    The output dict will contain:
    - 'image': a float tensor of shape [..., C, H, W]
    - 'label': a float tensor of shape [..., D, H, W]
    - 'weight': a float tensor of shape [..., 1, H, W]

    Args:
        stacked: A tensor with image, label, and weight concatenated along
          dim=-3.

    Returns:
        Output dictionary containing image, label, and weight tensors.
    """
    return {
        "image": stacked[..., : self.num_channels, :, :],
        "label": stacked[..., self.num_channels : -1, :, :],
        "weight": stacked[..., -1:, :, :],
    }


class SegmentationCutMixAndMixUp(nn.Module):
  """A CutMix and MixUp augmentation for segmentation tasks.

  Unlike the default 'CutMix' and 'MixUp' augmentations, which work for
  classification tasks, this version is designed for segmentation tasks.
  We assume the image+label+weights are concatenated along the channel
  dimension, and simply apply the mixing operations to the concatenated tensor.

  Since CutMix and MixUp assume an auxiliary label, we just add a dummy one and
  remove it.

  Requires an input tensor of shape [B, C, H, W] where 'B' is an even number.
  """

  def __init__(self, cutmix_alpha: float, mixup_alpha: float):
    """Initializes the augmentation module.

    If both cutmix_alpha and mixup_alpha are 0, no augmentation will be
    applied.
    If only one of the alpha values is greater than 0, only that
    augmentation will be applied.
    If both alpha values are greater than 0, one of the two augmentations will
    be selected at random with equal probability.

    Args:
      cutmix_alpha: The alpha parameter for the CutMix augmentation, or 0 to
        disable CutMix.f
      mixup_alpha: The alpha parameter for the MixUp augmentation, or 0 to
        disable MixUp.
    """

    super().__init__()
    transforms = []
    if cutmix_alpha > 0:
      transforms.append(torchvision_v2.CutMix(alpha=cutmix_alpha))
    if mixup_alpha > 0:
      transforms.append(torchvision_v2.MixUp(alpha=mixup_alpha))
    if len(transforms) == 1:
      self.transform = transforms[0]
    elif len(transforms) > 1:
      self.transform = torchvision.transforms.RandomChoice(transforms)
    else:
      self.transform = None

  def forward(self, stacked: torch.Tensor) -> torch.Tensor:
    if self.transform is None:
      return stacked
    dummy_label = torch.zeros(size=(stacked.shape[0], 1)).to(stacked.device)
    stacked, _ = self.transform(stacked, dummy_label)
    return stacked


class ApplyOnField(nn.Module):
  """A convenience module for applying a transformation on a specific field.

  This is useful when the dataset is defined as a dictionary, and we want to
  apply a transformation only on a specific field.

  For example, if the dataset has an 'image' field, and we want to apply a
  normalization on it.
  """

  def __init__(self, field: str, transform: nn.Module):
    super().__init__()
    self.field = field
    self.transform = transform

  def forward(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    res = {k: v for k, v in d.items() if k != self.field}
    res[self.field] = self.transform(d[self.field])
    return res


class ApplyOnList(nn.Module):
  """A convenience module for applying a transformation on list of samples."""

  def __init__(self, transform: nn.Module):
    super().__init__()
    self.transform = transform

  def forward(self, lst: list[Any]) -> list[Any]:
    return [self.transform(i) for i in lst]


class ValidationCheck(nn.Module):
  """A validation check on the shapes and values of the data.

  Image shape should be [..., C, H, W].
  Label shape should be [..., D, H, W].
  Weight shape should be [..., 1, H, W].

  Image RGB values should be in the range [-3, 3] (post normalisation).
  Label values should be in the range [0, 1], allowing a little error.
  Weight values should be in the range [0, 1], allowing a little error.
  """

  def __init__(self, *, num_channels: int = 3, num_classes: int = 0):
    """Initializes the validation check module.

    Args:
        num_channels: The number of channels in the image.
        num_classes: The number of classes in the label, or 0 to disable the
          label validation check.
    """

    super().__init__()
    self.num_channels = num_channels
    self.num_classes = num_classes

  def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    image, label, weight = x["image"], x["label"], x["weight"]

    def raise_if(condition, message):
      if condition:
        raise ValueError(message)

    raise_if(image.ndim < 3, f"Image dimension is too low ({image.ndim})")
    raise_if(label.ndim < 3, f"Label dimension is too low ({label.ndim})")
    raise_if(weight.ndim < 3, f"Weight dimension is too low ({weight.ndim})")
    outer = image.shape[:-3]
    height = image.shape[-2]
    width = image.shape[-1]
    raise_if(
        image.shape != (*outer, self.num_channels, height, width),
        f"Image shape {image.shape} does not match expected shape",
    )
    classes = self.num_classes if self.num_classes > 0 else label.shape[-3]
    raise_if(
        label.shape != (*outer, classes, height, width),
        f"Label shape {label.shape} does not match expected shape",
    )
    raise_if(
        weight.shape != (*outer, 1, height, width),
        f"Weight shape {weight.shape} does not match expected shape",
    )

    # Validate the image is properly normalized.
    raise_if(
        torch.any(image.lt(-3).bitwise_or(image.gt(3))),
        f"Image values out of range [-3, 3] ({image.min().item()} to"
        f" {image.max().item()})",
    )

    # Validate the label is in range [0, 1], up to a few errors, due to
    # interpolations.
    raise_if(
        torch.any(label.lt(-0.001).bitwise_or(label.gt(1.001))),
        f"Label values out of range [0, 1] ({label.min().item()} to"
        f" {label.max().item()})",
    )

    # Validate that the label probabilities sum to at most 1 (up to a few
    # errors). It could be less due to paddings.
    raise_if(
        torch.any(label.sum(axis=-3).gt(1.001)),
        "Label probabilities sum to more than 1"
        f" ({label.sum(axis=-3).max().item()})",
    )

    # Validate the weight is in range [0, 1], up to a few errors, due to
    # interpolations.
    raise_if(
        torch.any(weight.lt(-0.001).bitwise_or(weight.gt(1.001))),
        f"Weight values out of range [0, 1] ({weight.min().item()} to"
        f" {weight.max().item()})",
    )
    return x


class RandomColorJitter(nn.Module):
  """Applies standard torchvision ColorJitter (brightness, contrast, saturation, hue) to an RGB image."""

  def __init__(
      self,
      brightness: float = 0.2,
      contrast: float = 0.2,
      saturation: float = 0.2,
      hue: float = 0.1,
  ) -> None:
    """Initializes RandomColorJitter.

    Args:
        brightness: How much to jitter brightness (non-negative float).
        contrast: How much to jitter contrast (non-negative float).
        saturation: How much to jitter saturation (non-negative float).
        hue: How much to jitter hue (float between 0 and 0.5).
    """
    super().__init__()
    self.jitter = torchvision_v2.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

  def forward(self, image: Any) -> Any:
    """Applies torchvision ColorJitter to an image uint8 numpy array."""
    # Convert HWC uint8 numpy array to CHW uint8 tensor
    img_tensor = torch.from_numpy(image).permute(2, 0, 1)
    jittered_tensor = self.jitter(img_tensor)
    return jittered_tensor.permute(1, 2, 0).numpy()


class DetectionRandomFlipRotate(nn.Module):
  """Applies D4 dihedral group augmentations (hflip, vflip, transpose flip, rotate) to image and boxes."""

  def __init__(
      self,
      hflip_prob: float = 0.5,
      vflip_prob: float = 0.0,
      tflip_prob: float = 0.0,
      random_rotate: bool = False,
  ) -> None:
    """Initializes DetectionRandomFlipRotate.

    Args:
        hflip_prob: Probability of horizontal image flipping.
        vflip_prob: Probability of vertical image flipping.
        tflip_prob: Probability of transpose flipping (swapping X and Y axes).
        random_rotate: Whether to apply random 90-degree rotations.
    """
    super().__init__()
    self.hflip_prob = hflip_prob
    self.vflip_prob = vflip_prob
    self.tflip_prob = tflip_prob
    self.random_rotate = random_rotate

  def forward(
      self,
      image: Any,
      boxes_tensor: torch.Tensor,
      image_size: int,
  ) -> tuple[Any, torch.Tensor]:
    """Applies geometric flips, transpose, and rotations to image and bounding boxes.

    Args:
        image: Input image NumPy uint8 array of shape (H, W, C).
        boxes_tensor: Bounding boxes tensor of shape (N, 4) in [x1, y1, x2, y2]
          pixel coordinate format.
        image_size: Spatial dimension (width and height) of the image.

    Returns:
        Tuple of (augmented_image, augmented_boxes_tensor).
    """
    # Vertical flipping
    if self.vflip_prob > 0.0 and random.random() < self.vflip_prob:
      image = image[::-1, :, :].copy()
      if len(boxes_tensor) > 0:
        old_y1 = boxes_tensor[:, 1].clone()
        old_y2 = boxes_tensor[:, 3].clone()
        boxes_tensor[:, 1] = image_size - old_y2
        boxes_tensor[:, 3] = image_size - old_y1

    # Horizontal flipping
    if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
      image = image[:, ::-1, :].copy()
      if len(boxes_tensor) > 0:
        old_x1 = boxes_tensor[:, 0].clone()
        old_x2 = boxes_tensor[:, 2].clone()
        boxes_tensor[:, 0] = image_size - old_x2
        boxes_tensor[:, 2] = image_size - old_x1

    # Transpose flipping (swapping X and Y coordinates)
    if self.tflip_prob > 0.0 and random.random() < self.tflip_prob:
      image = image.transpose(1, 0, 2).copy()
      if len(boxes_tensor) > 0:
        boxes_tensor = boxes_tensor[:, [1, 0, 3, 2]]

    # Random 90-degree rotations
    if self.random_rotate:
      k = random.randint(0, 3)
      if k > 0:
        image = cv2.rotate(
            image,
            [
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ][k - 1],
        )

        if len(boxes_tensor) > 0:
          for _ in range(k):
            x1_old = boxes_tensor[:, 0].clone()
            y1_old = boxes_tensor[:, 1].clone()
            x2_old = boxes_tensor[:, 2].clone()
            y2_old = boxes_tensor[:, 3].clone()

            boxes_tensor[:, 0] = image_size - y2_old
            boxes_tensor[:, 1] = x1_old
            boxes_tensor[:, 2] = image_size - y1_old
            boxes_tensor[:, 3] = x2_old

    return image, boxes_tensor


# Backward-compatibility alias
DetectionRandomFlip = DetectionRandomFlipRotate
