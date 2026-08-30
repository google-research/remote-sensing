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

import tempfile
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from etils import epath
import numpy as np
from remote_sensing.models import visualizations
import torch
import torchvision


class FakeDetectionDataset:

  def __init__(self, num_samples=2):
    self.num_samples = num_samples

  def __len__(self):
    return self.num_samples

  def __getitem__(self, index):
    return torch.full((3, 32, 32), 0.5, dtype=torch.float32), dict()


class FakeDetectionModel(torch.nn.Module):

  def forward(self, images):
    preds = []
    for _ in images:
      preds.append({
          "boxes": torch.tensor([
              [10.0, 10.0, 20.0, 20.0],
              [0.0, 0.0, 5.0, 5.0],
          ]),
          "labels": torch.tensor([0, 1]),
          "scores": torch.tensor([0.9, 0.3]),
      })
    return preds


class VisualizationsTest(parameterized.TestCase):

  def test_denormalize(self):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    image = torch.rand((1, 3, 256, 256), dtype=torch.float32)
    normalize = torchvision.transforms.Normalize(mean, std)
    denormalize = visualizations.Denormalize(mean, std)
    norm_denorm = denormalize(normalize(image))
    np.testing.assert_allclose(
        norm_denorm.numpy(), image.numpy(), atol=1e-6, rtol=1e-6
    )

  @parameterized.parameters(
      dict(num_classes=2, min_dist=3**0.5),
      dict(num_classes=6, min_dist=1),
      dict(num_classes=17, min_dist=0.5),
      dict(num_classes=25, min_dist=0.5),
      dict(num_classes=62, min_dist=0.33),
  )
  def test_auto_label_palette(self, num_classes: int, min_dist: float):
    palette = visualizations.auto_label_palette(num_classes)
    self.assertEqual(palette.shape, (num_classes, 3))
    self.assertGreaterEqual(palette.min(), 0)
    self.assertLessEqual(palette.max(), 1)
    palette = palette.to(torch.float32)
    for i in range(num_classes):
      for j in range(i + 1, num_classes):
        self.assertGreaterEqual(
            torch.dist(palette[i], palette[j]),
            min_dist,
            f"Palette {palette[i]} and {palette[j]} are too close.",
        )

  @parameterized.parameters(
      dict(
          use_weights=False,
          golden_file="test_segmentation_golden_no_weights.png",
      ),
      dict(
          use_weights=True,
          golden_file="test_segmentation_golden_with_weights.png",
      ),
  )
  def test_visualize_segmentation(self, use_weights: bool, golden_file: str):
    resource = epath.resource_path("remote_sensing")

    image_file = resource / "(tests/testdata/test_image.png)"
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)  # uint8[H, W, BGR]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # float32[H, W, RGB]
    image = torch.from_numpy(image).permute(2, 0, 1)  # float32[RGB, H, W]
    image = image[None, :, :, :]  # float32[B, RGB, H, W]

    # int64[B, H, W]
    label_file = resource / "(tests/testdata/test_segmentation_label.png)"
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)  # uint8[H, W]
    label = torch.from_numpy(label).to(torch.long)  # int64[H, W]
    # int64[C, H, W]
    label = torch.nn.functional.one_hot(label, num_classes=3).permute(2, 0, 1)
    label = label[None, :, :, :].to(torch.float32)  # float32[C, H, W]

    # int64[B, H, W]
    pred_file = resource / "(tests/testdata/test_segmentation_prediction.png)"
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)  # uint8[H, W]
    pred = torch.from_numpy(pred).to(torch.long)  # int64[H, W]
    # int64[C, H, W]
    pred = torch.nn.functional.one_hot(pred, num_classes=3).permute(2, 0, 1)
    pred = pred[None, :, :, :].to(torch.float32)  # float32[B, C, H, W]

    # float32[B, 1, H, W]
    if use_weights:
      weight_file = resource / "(tests/testdata/test_segmentation_weight.png)"
      weight = cv2.imread(weight_file, cv2.IMREAD_GRAYSCALE)  # uint8[H, W]
      weight = torch.from_numpy(weight).to(torch.float32)  # float32[H, W]
      weight = weight[None, None, :, :]  # float32[B, 1, H, W]
    else:
      weight = None

    # float32[RGB, H', W']
    out = visualizations.visualize_segmentation(
        image,
        label,
        pred,
        weight,
        config=visualizations.SegmentationVisualizationConfig(
            label_palette=torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]),
        ),
    )
    # uint8[H', W', RGB]
    out = (out.permute(1, 2, 0) * 255).to(torch.uint8).numpy()

    golden_file = resource / "models/testdata" / golden_file
    golden = cv2.imread(golden_file, cv2.IMREAD_COLOR)  # uint8[H', W', BGR]
    golden = cv2.cvtColor(golden, cv2.COLOR_BGR2RGB)  # uint8[H', W', RGB]

    if not np.array_equal(out, golden):
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as outfile:
        cv2.imwrite(outfile.name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        self.fail(f"Output stored in {outfile.name} does not match golden.")

  def test_visualize_dataset_predictions_invalid_args(self):
    model = FakeDetectionModel()
    dataset = FakeDetectionDataset()
    config = visualizations.ObjectDetectionVisualizationConfig(
        class_names=["tree"]
    )
    with self.assertRaisesRegex(
        ValueError, "num_images must be a positive integer"
    ):
      visualizations.visualize_dataset_predictions(
          model, dataset, config, torch.device("cpu"), num_images=0
      )

    with self.assertRaisesRegex(
        ValueError, "dataset must contain at least one sample"
    ):
      visualizations.visualize_dataset_predictions(
          model, FakeDetectionDataset(0), config, torch.device("cpu")
      )

  def test_visualize_dataset_predictions(self):
    resource = epath.resource_path("remote_sensing")
    model = FakeDetectionModel()
    dataset = FakeDetectionDataset(num_samples=3)
    config = visualizations.ObjectDetectionVisualizationConfig(
        class_names=["tree", "house"],
        score_threshold=0.5,
    )
    fig = visualizations.visualize_dataset_predictions(
        model=model,
        dataset=dataset,
        config=config,
        device=torch.device("cpu"),
        num_images=2,
        seed=42,
    )
    self.assertIsNotNone(fig)
    self.assertLen(fig.axes, 2)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    visualizations.plt.close(fig)

    golden_file = (
        resource / "(tests/testdata/test_object_detection_golden.png)"
    )
    golden = cv2.imread(golden_file, cv2.IMREAD_COLOR)  # uint8[H', W', BGR]
    golden = cv2.cvtColor(golden, cv2.COLOR_BGR2RGB)  # uint8[H', W', RGB]

    if not np.array_equal(out, golden):
      diff = (
          np.max(np.abs(out.astype(int) - golden.astype(int)))
          if out.shape == golden.shape
          else -1
      )
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as outfile:
        cv2.imwrite(outfile.name, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        self.fail(
            f"Output stored in {outfile.name} does not match golden."
            f" diff={diff} out={out.shape} golden={golden.shape}"
        )

  def test_visualize_dataset_predictions_with_auto_label_palette(self):
    model = FakeDetectionModel()
    dataset = FakeDetectionDataset(num_samples=2)
    palette = cast(
        list[tuple[float, ...]],
        visualizations.auto_label_palette(2).tolist(),
    )
    config = visualizations.ObjectDetectionVisualizationConfig(
        class_names=["tree", "house"],
        class_colors=palette,
        score_threshold=0.5,
    )
    fig = visualizations.visualize_dataset_predictions(
        model=model,
        dataset=dataset,
        config=config,
        device=torch.device("cpu"),
        num_images=1,
    )
    self.assertIsNotNone(fig)
    visualizations.plt.close(fig)

  def test_visualize_dataset_predictions_with_denormalize(self):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denormalize = visualizations.Denormalize(mean, std)
    model = FakeDetectionModel()
    dataset = FakeDetectionDataset(num_samples=1)
    config = visualizations.ObjectDetectionVisualizationConfig(
        class_names=["tree", "house"],
    )
    fig = visualizations.visualize_dataset_predictions(
        model=model,
        dataset=dataset,
        config=config,
        device=torch.device("cpu"),
        num_images=1,
        denormalize=denormalize,
    )
    self.assertIsNotNone(fig)
    visualizations.plt.close(fig)


if __name__ == "__main__":
  absltest.main()
