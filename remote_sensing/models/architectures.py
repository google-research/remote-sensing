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

"""A collection of architectures for remote sensing tasks.

This module currently contains a UNet-style architecture for semantic
segmentation with ViT encoders.
"""

import collections
import torch
from torch import nn
import torch.nn.functional as F


OrderedDict = collections.OrderedDict


class ConvBlock(nn.Module):
  """A basic convolutional block.

  This block consists of two KxK convolutions (where K is kernel_size), each
  followed by GroupNorm and ReLU.
  """

  def __init__(
      self, in_c: int, out_c: int, groups: int = 8, kernel_size: int = 3
  ):
    """Initializes the ConvBlock.

    Args:
        in_c: Number of input channels.
        out_c: Number of output channels.
        groups: Number of groups for GroupNorm.
        kernel_size: Size of the convolutional kernels.
    """
    super().__init__()

    if in_c <= 0 or out_c <= 0:
      raise ValueError("in_c and out_c must be positive integers.")
    if groups <= 0:
      raise ValueError("groups must be a positive integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
      raise ValueError("kernel_size must be a positive odd integer.")

    if out_c % groups != 0:
      raise ValueError("out_c must be divisible by groups.")

    self.block: nn.Sequential = nn.Sequential(
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(groups, out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_c,
            out_c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.GroupNorm(groups, out_c),
        nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the convolutional block.

    Args:
        x: Input tensor.

    Returns:
        Output tensor.
    """
    return self.block(x)


class ViTUNetDecoder(nn.Module):
  """A UNet-style decoder for Vision Transformer features.

  This decoder takes a list of features from different encoder stages,
  progressively upsamples them, and combines them with skip connections
  to produce a dense prediction map.

  In a combined ViT + ViTUNetDecoder model the ViT patch size should be a
  power of 2 (i.e., 2^n), and the length of `decoder_channels` should usually
  be n + 1.
  """

  def __init__(
      self,
      encoder_dim: int,
      decoder_channels: tuple[int, ...] = (512, 256, 128, 64),
      output_dims: int = 1,
  ):
    super().__init__()

    if not decoder_channels:
      raise ValueError("decoder_channels must not be empty.")
    if output_dims <= 0:
      raise ValueError("output_dims must be a positive integer.")

    self.center = ConvBlock(encoder_dim, decoder_channels[0])

    up_blocks = []
    conv_blocks = []
    for i in range(len(decoder_channels) - 1):
      up_blocks.append(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
      )
      conv_blocks.append(
          ConvBlock(decoder_channels[i] + encoder_dim, decoder_channels[i + 1])
      )

    self.up_blocks = nn.ModuleList(up_blocks)
    self.conv_blocks = nn.ModuleList(conv_blocks)

    self.refine = ConvBlock(decoder_channels[-1], decoder_channels[-1])
    self.pred_head = nn.Conv2d(
        decoder_channels[-1], output_dims, kernel_size=1
    )

  def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    """Forward pass for the decoder.

    Args:
        features: List of features from encoder layers. Expects N tensors of
          shape (B, C, H, W), where B is batch size, C is encoder dimension,
          and H, W are spatial dimensions. Features are expected to be ordered
          from shallowest to deepest layers.

    Returns:
        Segmentation map logits.
    """
    if len(features) != len(self.conv_blocks) + 1:
      raise ValueError(
          f"Expected {len(self.conv_blocks) + 1} features, but"
          f" got {len(features)}."
      )

    x = self.center(features[-1])

    for i, (up, conv) in enumerate(zip(self.up_blocks, self.conv_blocks)):
      x = up(x)
      skip_feat = features[-2 - i]
      skip_feat = F.interpolate(
          skip_feat, size=x.shape[2:], mode="bilinear", align_corners=False
      )
      x = torch.cat([x, skip_feat], dim=1)
      x = conv(x)

    x = self.refine(x)
    return self.pred_head(x)


class LayerNorm2d(nn.Module):
  """LayerNorm applied over the channel dimension of a 2D image tensor."""

  def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
    """Initializes LayerNorm2d.

    Args:
        num_channels: Number of channels in the 2D input tensor.
        eps: Small epsilon value for numerical stability.
    """
    super().__init__()
    self.weight: nn.Parameter = nn.Parameter(torch.ones(num_channels))
    self.bias: nn.Parameter = nn.Parameter(torch.zeros(num_channels))
    self.eps: float = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Applies 2D layer normalization along the channel dimension.

    Args:
        x: Input 4D feature map tensor of shape (B, C, H, W), where B is batch
          size, C is the channel dimension (dim=1), H is height, and W is width.

    Returns:
        Normalized feature map tensor of shape (B, C, H, W).
    """
    mean = x.mean(dim=1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + self.eps)
    return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ConvLayerNormRPNHead(nn.Module):
  """Region Proposal Network (RPN) head with Conv2d layers and LayerNorm2d.

  This head processes multi-scale feature maps using N 3x3 convolutions with
  channel-wise LayerNorm2d and ReLU activations, followed by 1x1 convolutions
  for objectness classification logits and bounding-box regression offsets.
  """

  def __init__(
      self,
      in_channels: int = 256,
      num_anchors: int = 3,
      feat_channels: int = 256,
      num_convs: int = 2,
  ) -> None:
    """Initializes ConvLayerNormRPNHead.

    Args:
        in_channels: Number of input feature map channels (default: 256).
        num_anchors: Number of anchor boxes generated per spatial location.
        feat_channels: Number of intermediate convolutional channels (default:
          256).
        num_convs: Number of convolutional layers in the head (default: 2).
    """
    super().__init__()
    self.convs: nn.ModuleList = nn.ModuleList()
    curr_channels = in_channels
    for _ in range(num_convs):
      self.convs.append(
          nn.Sequential(
              nn.Conv2d(
                  curr_channels, feat_channels, kernel_size=3, padding=1
              ),
              LayerNorm2d(feat_channels),
              nn.ReLU(inplace=True),
          )
      )
      curr_channels = feat_channels

    self.cls_logits: nn.Conv2d = nn.Conv2d(
        feat_channels, num_anchors, kernel_size=1
    )
    self.bbox_pred: nn.Conv2d = nn.Conv2d(
        feat_channels, num_anchors * 4, kernel_size=1
    )

  def forward(
      self, x: list[torch.Tensor]
  ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logits = []
    bbox_reg = []
    for feature in x:
      h = feature
      for conv in self.convs:
        h = conv(h)
      logits.append(self.cls_logits(h))
      bbox_reg.append(self.bbox_pred(h))
    return logits, bbox_reg


class FastRCNNConvLayerNormBoxHead(nn.Module):
  """Fast R-CNN box head with Conv2d layers, LayerNorm2d, and 1 Linear layer.

  This head processes RoI-pooled feature maps using N 3x3 convolutions with
  channel-wise LayerNorm2d and ReLU activations, followed by a flattened linear
  projection layer for final classification and box regression.
  """

  def __init__(
      self,
      in_channels: int = 256,
      feat_channels: int = 256,
      num_convs: int = 4,
      roi_output_size: int = 7,
      fc_dims: int = 1024,
  ) -> None:
    """Initializes FastRCNNConvLayerNormBoxHead.

    Args:
        in_channels: Number of input channels from RoI pooling (default: 256).
        feat_channels: Number of intermediate convolutional channels (default:
          256).
        num_convs: Number of convolutional layers in the head (default: 4).
        roi_output_size: Spatial output size (height and width) from RoI pooling
          (default: 7, following Fast/Faster R-CNN standards:
          https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
          and
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
          ).
        fc_dims: Dimensionality of the output fully-connected representation
          (default: 1024).
    """
    super().__init__()
    self.convs: nn.ModuleList = nn.ModuleList()
    curr_channels = in_channels
    for _ in range(num_convs):
      self.convs.append(
          nn.Sequential(
              nn.Conv2d(
                  curr_channels,
                  feat_channels,
                  kernel_size=3,
                  padding=1,
                  bias=False,
              ),
              LayerNorm2d(feat_channels),
              nn.ReLU(inplace=True),
          )
      )
      curr_channels = feat_channels

    # Flatten spatial feature map
    # (feat_channels x roi_output_size x roi_output_size).
    # Default 7x7 spatial resolution follows Fast/Faster R-CNN standards:
    # - Paper:
    # https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    # - Code:
    # https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
    in_features = feat_channels * roi_output_size * roi_output_size
    self.fc: nn.Sequential = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features, fc_dims),
        nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = x
    for conv in self.convs:
      h = conv(h)
    return self.fc(h)


# Backward-compatibility alias
ConvLayerNormBoxHead = FastRCNNConvLayerNormBoxHead


class ViTBackbonePyramid(nn.Module):
  """A Vision Transformer (ViT) backbone with a single-feature spatial pyramid.

  The final ViT patch-token representation is reshaped into a spatial
  feature map. Additional feature levels are produced using interpolation
  and pooling for use with Torchvision detection models.
  """

  def __init__(
      self,
      encoder: nn.Module,
      image_size: int,
      out_channels: int = 256,
      freeze_encoder: bool = False,
  ):
    """Initializes the ViT backbone pyramid.

    Args:
        encoder: Pretrained Vision Transformer encoder module.
        image_size: Input image resolution.
        out_channels: Number of channels in each pyramid feature map.
        freeze_encoder: Whether to freeze the pretrained ViT encoder.
    """
    super().__init__()

    self.encoder: nn.Module = encoder

    self.image_size: int = image_size
    self.patch_size: int = self.encoder.config.patch_size  # pyrefly: ignore[bad-assignment]
    self.hidden_size: int = self.encoder.config.hidden_size  # pyrefly: ignore[bad-assignment]

    if self.patch_size not in (8, 16, 32):
      raise ValueError(
          f"Expected patch size in (8, 16, 32), got {self.patch_size}."
      )

    if self.image_size % self.patch_size != 0:
      raise ValueError(
          f"image_size={self.image_size} must be divisible by "
          f"patch_size={self.patch_size}."
      )

    self.projection: nn.Sequential = nn.Sequential(
        nn.Conv2d(
            self.hidden_size,
            out_channels,
            kernel_size=1,
        ),
        nn.GroupNorm(32, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ),
        nn.GroupNorm(32, out_channels),
        nn.ReLU(inplace=True),
    )

    # Required by Torchvision detection models.
    self.out_channels: int = out_channels

    if freeze_encoder:
      for parameter in self.encoder.parameters():
        parameter.requires_grad = False

  def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Extracts the multi-scale ViT feature pyramid.

    Args:
        images: Input tensor with shape (B, C, H, W).

    Returns:
        Ordered dictionary containing feature maps at strides 4, 8, 16, 32, and
        64 relative to the input image spatial resolution (H, W).
    """
    encoder_output = self.encoder(images)
    tokens = encoder_output.last_hidden_state

    batch_size, num_tokens, hidden_dim = tokens.shape
    # The input spatial dimensions (height, width) directly determine the
    # resolution of each pyramid level, where level P_k has spatial stride 2^k
    # (i.e. dimensions height // 2^k and width // 2^k relative to the image).
    height, width = images.shape[-2:]

    patch_height = height // self.patch_size
    patch_width = width // self.patch_size
    expected_tokens = patch_height * patch_width

    if num_tokens != expected_tokens:
      raise ValueError(
          f"Unexpected token count: received {num_tokens}, "
          f"expected {expected_tokens} "
          f"({patch_height} x {patch_width})."
      )

    features = tokens.reshape(
        batch_size,
        patch_height,
        patch_width,
        hidden_dim,
    )

    features = features.permute(0, 3, 1, 2).contiguous()
    features = self.projection(features)

    if self.patch_size == 16:
      # For patch_size=16, the base feature map has spatial stride 16 (P4).
      # We define P4 first, then derive P3 and P2 via 2x upsampling, and P5 and
      # P6 via 2x max pooling.
      p4 = features
      p3 = F.interpolate(
          p4, scale_factor=2, mode="bilinear", align_corners=False
      )
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)
    elif self.patch_size == 8:
      # For patch_size=8, the base feature map has spatial stride 8 (level P3).
      # We derive P2 via upsampling, and P4, P5, and P6 via max pooling.
      p3 = features
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p4 = F.max_pool2d(p3, kernel_size=2, stride=2)
      p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)
    else:  # self.patch_size == 32
      # For patch_size=32, the base feature map has spatial stride 32 (level
      # P5). We derive P4, P3, and P2 via upsampling, and P6 via max pooling.
      p5 = features
      p4 = F.interpolate(
          p5, scale_factor=2, mode="bilinear", align_corners=False
      )
      p3 = F.interpolate(
          p4, scale_factor=2, mode="bilinear", align_corners=False
      )
      p2 = F.interpolate(
          p3, scale_factor=2, mode="bilinear", align_corners=False
      )
      p6 = F.max_pool2d(p5, kernel_size=2, stride=2)

    return OrderedDict({
        "0": p2,
        "1": p3,
        "2": p4,
        "3": p5,
        "4": p6,
    })
