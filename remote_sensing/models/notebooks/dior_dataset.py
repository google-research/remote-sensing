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

"""DIOR dataset adapter for Faster R-CNN.

This module loads DIOR images with YOLO-format annotations and converts
normalized YOLO boxes into the absolute corner-coordinate format expected
by Torchvision detection models.
"""

import pathlib
import random
from typing import Any

import cv2
import torch
from torch.utils import data

Dataset = data.Dataset
Path = pathlib.Path


# ============================================================
# DIOR class definitions
# ============================================================

CLASS_NAMES = [
    "__background__",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# DIOR YOLO dataset adapter for Faster R-CNN
# ============================================================


class DIORYOLODataset(Dataset):
  """Load DIOR images and YOLO-format bounding-box annotations.

  YOLO annotations use normalized coordinates:

      class_id, center_x, center_y, width, height

  They are converted to the absolute corner-coordinate format required
  by Faster R-CNN:

      x_min, y_min, x_max, y_max
  """

  def __init__(
      self,
      img_dir: str | Path,
      label_dir: str | Path,
      image_size: int,
      train: bool = False,
      hflip_prob: float = 0.5,
      expected_images: int | None = None,
  ) -> None:
    """Initializes the DIOR dataset adapter.

    Args:
        img_dir: Path to the directory containing JPG images.
        label_dir: Path to the directory containing YOLO text annotations.
        image_size: Target size in pixels for resizing images and bounding
          boxes.
        train: Whether to apply training augmentations (e.g. horizontal
          flipping).
        hflip_prob: Probability of flipping the image horizontally during
          training.
        expected_images: Optional expected count of images for validation.

    Raises:
        FileNotFoundError: If the image or label directory does not exist.
        RuntimeError: If no JPG images are found, or if the count does not
          match expected_images.
    """
    self.img_dir = Path(img_dir)
    self.label_dir = Path(label_dir)
    self.image_size = image_size
    self.train = train
    self.hflip_prob = hflip_prob

    if not self.img_dir.exists():
      raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")

    if not self.label_dir.exists():
      raise FileNotFoundError(
          f"Label directory does not exist: {self.label_dir}"
      )

    # Sort image paths to preserve deterministic sample ordering.
    self.images = sorted(self.img_dir.glob("*.jpg"))

    if not self.images:
      raise RuntimeError(f"No JPG images were found in: {self.img_dir}")

    if expected_images is not None and len(self.images) != expected_images:
      raise RuntimeError(
          f"Expected {expected_images} images in {self.img_dir}, "
          f"but found {len(self.images)}."
      )

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(
      self,
      index: int,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Retrieve a single image and its corresponding detection targets.

    Args:
        index: Index of the image sample in the dataset.

    Returns:
        A tuple of (image_tensor, target_dict).

    Raises:
        RuntimeError: If the image cannot be read by OpenCV.
        ValueError: If a YOLO annotation line is invalid or has an unknown
          class ID.
    """
    img_path = self.images[index]
    label_path = self.label_dir / f"{img_path.stem}.txt"

    # OpenCV reads images in BGR channel order.
    image = cv2.imread(str(img_path))

    if image is None:
      raise RuntimeError(f"Could not read image: {img_path}")

    # Convert BGR to RGB and resize to the detector input resolution.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (self.image_size, self.image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    boxes: list[list[float]] = []
    labels: list[int] = []

    # We derive label_path from img_path.stem rather than looking it up by
    # index in a list of label files. This ensures robust pairing even if
    # some images lack annotation files (which are correctly treated as
    # background-only images without bounding boxes).
    if label_path.exists():
      for line_number, line in enumerate(
          label_path.read_text(encoding="utf-8").splitlines(),
          start=1,
      ):
        line = line.strip()

        if not line:
          continue

        values = line.split()

        if len(values) != 5:
          raise ValueError(
              f"Invalid YOLO annotation in {label_path} "
              f"at line {line_number}: expected 5 values, "
              f"found {len(values)}."
          )

        cls, center_x, center_y, box_width, box_height = map(
            float,
            values,
        )

        cls = int(cls)

        if not 0 <= cls < NUM_CLASSES - 1:
          raise ValueError(
              f"Invalid class ID {cls} in {label_path}. "
              f"Expected a value between 0 and {NUM_CLASSES - 2}."
          )

        # Convert normalized YOLO coordinates to absolute corners.
        x1 = (center_x - box_width / 2.0) * self.image_size
        y1 = (center_y - box_height / 2.0) * self.image_size
        x2 = (center_x + box_width / 2.0) * self.image_size
        y2 = (center_y + box_height / 2.0) * self.image_size

        # Clip coordinates to the resized image boundaries.
        x1 = max(0.0, min(float(self.image_size), x1))
        y1 = max(0.0, min(float(self.image_size), y1))
        x2 = max(0.0, min(float(self.image_size), x2))
        y2 = max(0.0, min(float(self.image_size), y2))

        # Skip invalid or zero-area boxes.
        if x2 <= x1 or y2 <= y1:
          continue

        boxes.append([x1, y1, x2, y2])

        # Faster R-CNN reserves class index 0 for background.
        labels.append(cls + 1)

    # Faster R-CNN requires boxes to have shape [N, 4], including
    # background-only images where N equals zero.
    boxes_tensor = torch.as_tensor(
        boxes,
        dtype=torch.float32,
    ).reshape(-1, 4)

    labels_tensor = torch.as_tensor(
        labels,
        dtype=torch.int64,
    )

    # Apply horizontal flipping during training and transform boxes
    # consistently with the image.
    if self.train and random.random() < self.hflip_prob:
      image = image[:, ::-1, :].copy()

      if len(boxes_tensor) > 0:
        old_x1 = boxes_tensor[:, 0].clone()
        old_x2 = boxes_tensor[:, 2].clone()

        boxes_tensor[:, 0] = self.image_size - old_x2
        boxes_tensor[:, 2] = self.image_size - old_x1

    # Convert HWC uint8 image into a CHW float tensor in [0, 1].
    image_tensor = (
        torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    )

    if len(boxes_tensor) > 0:
      area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
          boxes_tensor[:, 3] - boxes_tensor[:, 1]
      )
    else:
      area = torch.empty((0,), dtype=torch.float32)

    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor([index], dtype=torch.int64),
        "area": area,
        "iscrowd": torch.zeros(
            (len(boxes_tensor),),
            dtype=torch.int64,
        ),
    }

    return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
  """Preserve images and targets as separate tuples.

  Detection samples contain different numbers of objects and therefore
  cannot be stacked using PyTorch's default collation behavior.

  Args:
      batch: List of (image, target) tuples.

  Returns:
      A tuple containing (images_tuple, targets_tuple).
  """

  return tuple(zip(*batch))
