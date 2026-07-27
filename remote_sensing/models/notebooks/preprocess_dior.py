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

"""Prepare the DIOR dataset for YOLO-style loading.

This script:
1. Extracts the nested DIOR annotation archives when needed.
2. Organizes train and validation image directories.
3. Converts horizontal XML bounding boxes to YOLO text labels.
4. Creates data.yaml.
5. Verifies the resulting dataset structure.
"""

import pathlib
import shutil
import xml.etree.ElementTree as ET
import zipfile

Path = pathlib.Path


# ============================================================
# Dataset configuration
# ============================================================

DIOR_DIR = Path("datasets/DIOR")
METADATA_DIR = DIOR_DIR / "DIOR"

ANNOTATIONS_ARCHIVE = METADATA_DIR / "Annotations.zip"
IMAGESETS_ARCHIVE = METADATA_DIR / "ImageSets.zip"

ANNOTATIONS_DIR = METADATA_DIR / "Annotations" / "Horizontal Bounding Boxes"

MAIN_DIR = METADATA_DIR / "Main"

SOURCE_TRAINVAL_IMAGES = DIOR_DIR / "JPEGImages-trainval"
SOURCE_TEST_IMAGES = DIOR_DIR / "JPEGImages-test"

TRAIN_IMAGE_DIR = DIOR_DIR / "train" / "images"
TRAIN_LABEL_DIR = DIOR_DIR / "train" / "labels"

VALID_IMAGE_DIR = DIOR_DIR / "valid" / "images"
VALID_LABEL_DIR = DIOR_DIR / "valid" / "labels"

CATEGORIES = [
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

CLASS_TO_INDEX = {category: index for index, category in enumerate(CATEGORIES)}


# ============================================================
# Archive extraction
# ============================================================


def extract_if_needed(
    archive_path: Path,
    expected_output: Path,
    destination: Path,
) -> None:
  """Extract a ZIP archive only when its expected output is missing.

  Args:
      archive_path: Path to the ZIP archive.
      expected_output: Path to the expected output directory or file.
      destination: Path where the archive should be extracted.

  Raises:
      FileNotFoundError: If the archive path does not exist.
  """

  if expected_output.exists():
    print(f"Already extracted: {expected_output}")
    return

  if not archive_path.exists():
    raise FileNotFoundError(f"Archive not found: {archive_path}")

  print(f"Extracting {archive_path} into {destination}")

  with zipfile.ZipFile(archive_path, "r") as archive:
    archive.extractall(destination)


# ============================================================
# Image organization
# ============================================================


def prepare_image_directory(
    source_directory: Path,
    destination_directory: Path,
) -> None:
  """Move an extracted image directory to the expected dataset structure.

  The operation is safe to rerun.

  Args:
      source_directory: Path to the source image directory.
      destination_directory: Path to the destination image directory.

  Raises:
      FileNotFoundError: If the source directory does not exist.
      RuntimeError: If the destination directory exists and is not empty.
  """

  existing_images = list(destination_directory.glob("*.jpg"))

  if existing_images:
    print(
        f"Images already available at {destination_directory}: "
        f"{len(existing_images)}"
    )
    return

  if not source_directory.exists():
    raise FileNotFoundError(
        f"Image source not found: {source_directory}\n"
        f"Expected destination: {destination_directory}"
    )

  destination_directory.parent.mkdir(
      parents=True,
      exist_ok=True,
  )

  if destination_directory.exists():
    if any(destination_directory.iterdir()):
      raise RuntimeError(
          f"Destination exists and is not empty: {destination_directory}"
      )

    destination_directory.rmdir()

  shutil.move(
      str(source_directory),
      str(destination_directory),
  )

  print(f"Moved {source_directory} to {destination_directory}")


# ============================================================
# Annotation conversion
# ============================================================


def convert_bbox(
    image_width: int,
    image_height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float]:
  """Convert an absolute XML bounding box to normalized YOLO format.

  Args:
      image_width: Width of the image in pixels.
      image_height: Height of the image in pixels.
      xmin: Minimum X coordinate of the bounding box.
      ymin: Minimum Y coordinate of the bounding box.
      xmax: Maximum X coordinate of the bounding box.
      ymax: Maximum Y coordinate of the bounding box.

  Returns:
      A tuple of (center_x, center_y, width, height) in normalized coordinates.
  """

  xmin = max(0.0, min(float(image_width), xmin))
  ymin = max(0.0, min(float(image_height), ymin))
  xmax = max(0.0, min(float(image_width), xmax))
  ymax = max(0.0, min(float(image_height), ymax))

  center_x = ((xmin + xmax) / 2.0) / image_width
  center_y = ((ymin + ymax) / 2.0) / image_height
  box_width = (xmax - xmin) / image_width
  box_height = (ymax - ymin) / image_height

  return center_x, center_y, box_width, box_height


def read_split_ids(split_file: Path) -> list[str]:
  """Read image identifiers from a DIOR split file.

  Args:
      split_file: Path to the split text file.

  Returns:
      A list of image identifier strings.

  Raises:
      FileNotFoundError: If the split file does not exist.
  """

  if not split_file.exists():
    raise FileNotFoundError(f"Split file not found: {split_file}")

  return [
      line.strip()
      for line in split_file.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]


def process_split(
    split_file: Path,
    destination_label_dir: Path,
) -> list[str]:
  """Convert one DIOR XML split into YOLO label files.

  Args:
      split_file: Path to the split text file containing image IDs.
      destination_label_dir: Path to the directory where YOLO labels will be
        saved.

  Returns:
      A list of processed image identifier strings.

  Raises:
      ValueError: If an XML file has missing or invalid size or class
      information.
      RuntimeError: If any required XML annotations are missing.
  """

  image_ids = read_split_ids(split_file)

  destination_label_dir.mkdir(
      parents=True,
      exist_ok=True,
  )

  converted_count = 0
  object_count = 0
  invalid_box_count = 0
  missing_annotations = []

  print(f"\nProcessing {len(image_ids)} images from {split_file.name}")

  for image_id in image_ids:
    xml_path = ANNOTATIONS_DIR / f"{image_id}.xml"
    label_path = destination_label_dir / f"{image_id}.txt"

    if not xml_path.exists():
      missing_annotations.append(image_id)
      continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")

    if size_node is None:
      raise ValueError(f"Missing image size in: {xml_path}")

    width_node = size_node.find("width")
    height_node = size_node.find("height")

    if width_node is None or height_node is None:
      raise ValueError(f"Incomplete image size in: {xml_path}")

    image_width = int(width_node.text)  # pyrefly: ignore[bad-argument-type]
    image_height = int(height_node.text)  # pyrefly: ignore[bad-argument-type]

    if image_width <= 0 or image_height <= 0:
      raise ValueError(f"Invalid image dimensions in: {xml_path}")

    yolo_annotations = []

    for obj in root.findall("object"):
      name_node = obj.find("name")
      bbox_node = obj.find("bndbox")

      if name_node is None or bbox_node is None:
        continue

      raw_name = name_node.text
      if raw_name is None:
        continue
      class_name = raw_name.strip()

      if class_name not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown DIOR class '{class_name}' in {xml_path}")

      xmin_node = bbox_node.find("xmin")
      ymin_node = bbox_node.find("ymin")
      xmax_node = bbox_node.find("xmax")
      ymax_node = bbox_node.find("ymax")

      if (
          xmin_node is None
          or ymin_node is None
          or xmax_node is None
          or ymax_node is None
      ):
        invalid_box_count += 1
        continue

      raw_xmin = xmin_node.text
      raw_ymin = ymin_node.text
      raw_xmax = xmax_node.text
      raw_ymax = ymax_node.text

      if (
          raw_xmin is None
          or raw_ymin is None
          or raw_xmax is None
          or raw_ymax is None
      ):
        invalid_box_count += 1
        continue

      xmin = float(raw_xmin)
      ymin = float(raw_ymin)
      xmax = float(raw_xmax)
      ymax = float(raw_ymax)

      if xmax <= xmin or ymax <= ymin:
        invalid_box_count += 1
        continue

      (
          center_x,
          center_y,
          box_width,
          box_height,
      ) = convert_bbox(
          image_width=image_width,
          image_height=image_height,
          xmin=xmin,
          ymin=ymin,
          xmax=xmax,
          ymax=ymax,
      )

      if box_width <= 0 or box_height <= 0:
        invalid_box_count += 1
        continue

      class_id = CLASS_TO_INDEX[class_name]

      yolo_annotations.append(
          f"{class_id} "
          f"{center_x:.6f} "
          f"{center_y:.6f} "
          f"{box_width:.6f} "
          f"{box_height:.6f}"
      )

      object_count += 1

    label_content = "\n".join(yolo_annotations)

    if label_content:
      label_content += "\n"

    label_path.write_text(
        label_content,
        encoding="utf-8",
    )

    converted_count += 1

  print(f"Converted labels: {converted_count}")
  print(f"Objects converted: {object_count}")
  print(f"Invalid boxes skipped: {invalid_box_count}")
  print(f"Missing annotations: {len(missing_annotations)}")

  if missing_annotations:
    raise RuntimeError(
        f"{len(missing_annotations)} annotations were missing. "
        f"Examples: {missing_annotations[:5]}"
    )

  return image_ids


# ============================================================
# Metadata and validation
# ============================================================


def create_data_yaml() -> Path:
  """Create the YOLO dataset metadata file.

  Returns:
      Path to the created data.yaml file.
  """

  yaml_lines = [
      "path: datasets/DIOR",
      "train: train/images",
      "val: valid/images",
      "",
      "names:",
  ]

  yaml_lines.extend(
      f"  {index}: {category}" for index, category in enumerate(CATEGORIES)
  )

  yaml_path = DIOR_DIR / "data.yaml"

  yaml_path.write_text(
      "\n".join(yaml_lines) + "\n",
      encoding="utf-8",
  )

  return yaml_path


def verify_dataset(
    trainval_ids: list[str],
    test_ids: list[str],
) -> None:
  """Verify that all expected images and labels were created.

  Args:
      trainval_ids: List of expected training and validation image IDs.
      test_ids: List of expected test image IDs.

  Raises:
      RuntimeError: If any expected images or labels are missing or incorrect.
  """

  train_image_count = len(list(TRAIN_IMAGE_DIR.glob("*.jpg")))
  train_label_count = len(list(TRAIN_LABEL_DIR.glob("*.txt")))
  valid_image_count = len(list(VALID_IMAGE_DIR.glob("*.jpg")))
  valid_label_count = len(list(VALID_LABEL_DIR.glob("*.txt")))

  missing_train_images = [
      image_id
      for image_id in trainval_ids
      if not (TRAIN_IMAGE_DIR / f"{image_id}.jpg").exists()
  ]

  missing_valid_images = [
      image_id
      for image_id in test_ids
      if not (VALID_IMAGE_DIR / f"{image_id}.jpg").exists()
  ]

  print("\nPreprocessing summary")
  print(f"Training images: {train_image_count}")
  print(f"Training labels: {train_label_count}")
  print(f"Expected training samples: {len(trainval_ids)}")
  print(f"Validation images: {valid_image_count}")
  print(f"Validation labels: {valid_label_count}")
  print(f"Expected validation samples: {len(test_ids)}")

  if missing_train_images:
    raise RuntimeError(
        f"Missing {len(missing_train_images)} training images. "
        f"Examples: {missing_train_images[:5]}"
    )

  if missing_valid_images:
    raise RuntimeError(
        f"Missing {len(missing_valid_images)} validation images. "
        f"Examples: {missing_valid_images[:5]}"
    )

  if train_label_count != len(trainval_ids):
    raise RuntimeError(
        f"Expected {len(trainval_ids)} training labels, "
        f"but found {train_label_count}."
    )

  if valid_label_count != len(test_ids):
    raise RuntimeError(
        f"Expected {len(test_ids)} validation labels, "
        f"but found {valid_label_count}."
    )

  print("DIOR dataset structure verified successfully.")


# ============================================================
# Main execution
# ============================================================


def main() -> None:
  """Prepare the complete DIOR dataset.

  Raises:
      FileNotFoundError: If required DIOR directories or files are missing.
      RuntimeError: If no XML annotations are found.
  """

  extract_if_needed(
      archive_path=ANNOTATIONS_ARCHIVE,
      expected_output=ANNOTATIONS_DIR,
      destination=METADATA_DIR,
  )

  extract_if_needed(
      archive_path=IMAGESETS_ARCHIVE,
      expected_output=MAIN_DIR,
      destination=METADATA_DIR,
  )

  required_paths = [
      ANNOTATIONS_DIR,
      MAIN_DIR,
      MAIN_DIR / "train.txt",
      MAIN_DIR / "val.txt",
      MAIN_DIR / "test.txt",
  ]

  for path in required_paths:
    if not path.exists():
      raise FileNotFoundError(f"Required DIOR path not found: {path}")

  xml_count = len(list(ANNOTATIONS_DIR.glob("*.xml")))

  if xml_count == 0:
    raise RuntimeError(f"No XML annotations were found in: {ANNOTATIONS_DIR}")

  print(f"Horizontal XML annotations found: {xml_count}")

  prepare_image_directory(
      SOURCE_TRAINVAL_IMAGES,
      TRAIN_IMAGE_DIR,
  )

  prepare_image_directory(
      SOURCE_TEST_IMAGES,
      VALID_IMAGE_DIR,
  )

  train_ids = process_split(
      MAIN_DIR / "train.txt",
      TRAIN_LABEL_DIR,
  )

  validation_ids = process_split(
      MAIN_DIR / "val.txt",
      TRAIN_LABEL_DIR,
  )

  test_ids = process_split(
      MAIN_DIR / "test.txt",
      VALID_LABEL_DIR,
  )

  trainval_ids = train_ids + validation_ids

  yaml_path = create_data_yaml()
  print(f"\nMetadata file created: {yaml_path}")

  verify_dataset(
      trainval_ids=trainval_ids,
      test_ids=test_ids,
  )


if __name__ == "__main__":
  main()
