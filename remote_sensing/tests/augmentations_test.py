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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from remote_sensing.models import augmentations
import torch
import torchvision


class AugmentationsTest(parameterized.TestCase):

  @parameterized.parameters(((),), ((1,),), ((3,),), ((2, 2),))
  def test_prepare_segmentation_label_and_weight(self, batch_shape):
    x = {
        "image": torch.rand(size=(*batch_shape, 3, 10, 10)),
        "label": torch.randint(0, 17, size=(*batch_shape, 10, 10)),
    }
    output = augmentations.PrepareSegmentationLabelAndWeight(17)(x)
    self.assertEqual(output["image"].shape, (*batch_shape, 3, 10, 10))
    self.assertEqual(output["label"].shape, (*batch_shape, 17, 10, 10))
    self.assertEqual(output["weight"].shape, (*batch_shape, 1, 10, 10))

  @parameterized.parameters(((),), ((1,),), ((3,),), ((2, 2),))
  def test_stack_segmentation_image_label_weight(self, batch_shape):
    x = {
        "image": torch.rand(size=(*batch_shape, 3, 10, 10)),
        "label": torch.rand(size=(*batch_shape, 17, 10, 10)),
        "weight": torch.rand(size=(*batch_shape, 1, 10, 10)),
    }
    output = augmentations.StackSegmentationImageLabelWeight()(x).numpy()
    self.assertEqual(output.shape, (*batch_shape, 3 + 17 + 1, 10, 10))
    np.testing.assert_allclose(output[..., :3, :, :], x["image"].numpy())
    np.testing.assert_allclose(output[..., 3:-1, :, :], x["label"].numpy())
    np.testing.assert_allclose(output[..., -1:, :, :], x["weight"].numpy())

  @parameterized.parameters(((),), ((1,),), ((3,),), ((2, 2),))
  def test_split_segmentation_image_label_weight(self, batch_shape):
    x = torch.rand(size=(*batch_shape, 3 + 17 + 1, 10, 10))
    output = augmentations.SplitSegmentationImageLabelWeight()(x)
    x = x.numpy()
    self.assertEqual(output["image"].shape, (*batch_shape, 3, 10, 10))
    self.assertEqual(output["label"].shape, (*batch_shape, 17, 10, 10))
    self.assertEqual(output["weight"].shape, (*batch_shape, 1, 10, 10))
    np.testing.assert_allclose(output["image"].numpy(), x[..., :3, :, :])
    np.testing.assert_allclose(output["label"].numpy(), x[..., 3:-1, :, :])
    np.testing.assert_allclose(output["weight"].numpy(), x[..., -1:, :, :])

  def test_segmentation_cutmix_and_mixup(self):
    image_label1 = torch.ones(size=(1, 2, 10, 10))
    image_label2 = torch.zeros(size=(1, 2, 10, 10))
    x = torch.cat([image_label1, image_label2], dim=0)
    output = augmentations.SegmentationCutMixAndMixUp(
        cutmix_alpha=0.7, mixup_alpha=0.5
    )(x)
    self.assertEqual(output.shape, (2, 2, 10, 10))
    np.testing.assert_array_less(output.numpy(), 1.01)
    np.testing.assert_array_less(-output.numpy(), 0.01)
    np.testing.assert_allclose(torch.sum(output, dim=0).numpy(), 1)
    self.assertGreater(torch.sum(output[0, ...]).numpy(), 0)
    self.assertLess(torch.sum(output[0, ...]).numpy(), 200)

  def test_apply_on_field(self):
    x = {
        "image": torch.zeros(size=(1,)),
        "label": torch.full(size=(1,), fill_value=42),
    }
    output = augmentations.ApplyOnField(
        field="image",
        transform=lambda y: y + 1,
    )(x)
    self.assertEqual(output["image"].numpy(), np.ones((1,)))
    self.assertEqual(output["label"].numpy(), np.full((1,), fill_value=42))

  def test_validation_check_passes(self):
    x = {
        "image": torch.rand(size=(1, 3, 10, 10)) * 6 - 3,
        "label": torch.randint(0, 17, size=(1, 10, 10)),
    }
    x = augmentations.PrepareSegmentationLabelAndWeight(17)(x)
    output = augmentations.ValidationCheck(num_channels=3, num_classes=17)(x)
    self.assertEqual(output, x)

  def test_validation_check_fails(self):
    validation_check = augmentations.ValidationCheck(
        num_channels=3, num_classes=17
    )
    with self.assertRaisesRegex(ValueError, "Image dimension is too low"):
      validation_check({
          "image": torch.rand(size=(10, 10)),
          "label": torch.rand(size=(17, 10, 10)),
          "weight": torch.rand(size=(1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Label dimension is too low"):
      validation_check({
          "image": torch.rand(size=(3, 10, 10)),
          "label": torch.rand(size=(10, 10)),
          "weight": torch.rand(size=(1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Weight dimension is too low"):
      validation_check({
          "image": torch.rand(size=(3, 10, 10)),
          "label": torch.rand(size=(17, 10, 10)),
          "weight": torch.rand(size=(10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Image shape .* does not match"):
      validation_check({
          "image": torch.rand(size=(1, 4, 10, 10)),
          "label": torch.rand(size=(1, 17, 10, 10)),
          "weight": torch.rand(size=(1, 1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Label shape .* does not match"):
      validation_check({
          "image": torch.rand(size=(1, 3, 10, 10)),
          "label": torch.rand(size=(1, 18, 10, 10)),
          "weight": torch.rand(size=(1, 1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Label shape .* does not match"):
      validation_check({
          "image": torch.rand(size=(1, 3, 10, 10)),
          "label": torch.rand(size=(1, 17, 20, 10)),
          "weight": torch.rand(size=(1, 1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Weight shape .* does not match"):
      validation_check({
          "image": torch.rand(size=(1, 3, 10, 10)),
          "label": torch.rand(size=(1, 17, 10, 10)),
          "weight": torch.rand(size=(1, 2, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Weight shape .* does not match"):
      validation_check({
          "image": torch.rand(size=(1, 3, 10, 10)),
          "label": torch.rand(size=(1, 17, 10, 10)),
          "weight": torch.rand(size=(2, 1, 10, 10)),
      })
    with self.assertRaisesRegex(ValueError, "Image values out of range"):
      validation_check({
          "image": torch.randint(0, 256, size=(1, 3, 100, 100)),
          "label": torch.rand(size=(1, 17, 100, 100)),
          "weight": torch.rand(size=(1, 1, 100, 100)),
      })
    with self.assertRaisesRegex(ValueError, "Label values out of range"):
      validation_check({
          "image": torch.rand(size=(1, 3, 100, 100)),
          "label": torch.rand(size=(1, 17, 100, 100)) * 3 - 1,
          "weight": torch.rand(size=(1, 1, 100, 100)),
      })
    with self.assertRaisesRegex(
        ValueError, "Label probabilities sum to more than 1"
    ):
      validation_check({
          "image": torch.rand(size=(1, 3, 100, 100)),
          "label": torch.rand(size=(1, 17, 100, 100)),
          "weight": torch.rand(size=(1, 1, 100, 100)),
      })
    with self.assertRaisesRegex(ValueError, "Weight values out of range"):
      validation_check({
          "image": torch.rand(size=(1, 3, 100, 100)),
          "label": torch.zeros(size=(1, 17, 100, 100)),
          "weight": torch.rand(size=(1, 1, 100, 100)) * 3 - 1,
      })

  def test_end_to_end(self):
    x = {
        "image": torch.randint(0, 256, size=(10, 3, 100, 100)) / 255.0,
        "label": torch.randint(0, 17, size=(10, 100, 100)),
    }
    transforms = torchvision.transforms.Compose([
        augmentations.PrepareSegmentationLabelAndWeight(17),
        augmentations.ApplyOnField(
            "image",
            torchvision.transforms.v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
        ),
        augmentations.StackSegmentationImageLabelWeight(),
        torchvision.transforms.RandomResizedCrop(
            size=(50, 50),
            antialias=True,
            scale=(0.2, 1.0),
            ratio=(1.0, 1.0),
        ),
        torchvision.transforms.v2.RandomHorizontalFlip(),
        torchvision.transforms.v2.RandomRotation([-180, 180]),
        augmentations.SegmentationCutMixAndMixUp(
            cutmix_alpha=0.7, mixup_alpha=0.5
        ),
        augmentations.SplitSegmentationImageLabelWeight(),
        augmentations.ApplyOnField(
            "image",
            torchvision.transforms.Normalize(
                mean=torch.Tensor([0.485, 0.456, 0.406]),
                std=torch.Tensor([0.229, 0.224, 0.225]),
            ),
        ),
        augmentations.ValidationCheck(num_channels=3, num_classes=17),
    ])
    outputs = transforms(x)
    self.assertEqual(outputs["image"].shape, (10, 3, 50, 50))
    self.assertEqual(outputs["label"].shape, (10, 17, 50, 50))
    self.assertEqual(outputs["weight"].shape, (10, 1, 50, 50))


if __name__ == "__main__":
  absltest.main()
