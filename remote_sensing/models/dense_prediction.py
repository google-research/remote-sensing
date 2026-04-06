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

"""A ViT Dense Prediction model.

This module implements an encoder-decoder architecture for dense prediction
based on Vision Transformers (ViT). The encoder is a standard ViT model that
takes an image as input and outputs a sequence of patch embeddings. The decoder
is a transformer-based model that takes the patch embeddings as input and
outputs features for each patch. These features are then rearranged to form the
full dense prediction map.
"""

from collections.abc import Callable
from typing import Any
from remote_sensing.models import architectures
from remote_sensing.models import positional_embeddings
from remote_sensing.models import vits
import torch
from torch import nn
import transformers


class DecoderConfig(transformers.PretrainedConfig):
  """Configuration for a ViT-based dense prediction decoder."""

  model_type = "dense_prediction_decoder"

  def __init__(self, output_dims=0, encoder_hidden_size=0, **kwargs):
    super().__init__(**kwargs)
    self.encoder_hidden_size = encoder_hidden_size
    self.output_dims = output_dims


class VitDecoderConfig(DecoderConfig, transformers.ViTConfig):
  """Configuration for a ViT-based dense prediction decoder."""

  model_type = "vit_dense_prediction_decoder"

  def __init__(
      self, output_dims=0, encoder_hidden_size=0, vit_config=None, **kwargs
  ):
    super().__init__(output_dims, encoder_hidden_size, **kwargs)
    self.pos_emb_type = kwargs.get("pos_emb_type", "trainable")


class SkipUNetDecoderConfig(DecoderConfig):
  """Configuration for a UNet-based dense prediction decoder with skip connections."""

  model_type = "unet_dense_prediction_decoder"

  def __init__(
      self,
      output_dims: int = 0,
      encoder_hidden_size: int = 0,
      decoder_dims: tuple[int, ...] = (),
      skip_connections: tuple[int, ...] = (),
      **kwargs,
  ):
    super().__init__(output_dims, encoder_hidden_size, **kwargs)

    self.decoder_dims = decoder_dims
    self.skip_connections = skip_connections

    assert len(decoder_dims) == len(
        skip_connections
    ), "Decoder dimensions and skip connections must have the same length."


class EncoderDecoderConfig(transformers.PretrainedConfig):
  """Configuration for a ViT-based dense prediction model.

  Attributes:
      encoder_config: Configuration for the ViT encoder.
      decoder_config: Configuration for the dense prediction decoder.
  """

  model_type = "vit_dense_prediction_model"

  def __init__(
      self,
      encoder_config: transformers.ViTConfig | None = None,
      decoder_config: DecoderConfig | None = None,
      **kwargs,
  ):
    super().__init__(**kwargs)

    # Handle loading from dict. This is necessary to be able to load models
    # with `from_pretrained`.
    if isinstance(encoder_config, dict):
      self.encoder_config = transformers.ViTConfig(**encoder_config)
    else:
      self.encoder_config = encoder_config

    if isinstance(decoder_config, dict):
      self.decoder_config = VitDecoderConfig(**decoder_config)
    else:
      self.decoder_config = decoder_config

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    # This is necessary to be able to save models with `save_pretrained`.
    output = super().to_dict()
    if self.encoder_config:
      output["encoder_config"] = self.encoder_config.to_dict()
    if self.decoder_config:
      output["decoder_config"] = self.decoder_config.to_dict()
    return output


class ViTDecoder(transformers.PreTrainedModel):
  """A ViT dense prediction decoder.

  The decoder takes preprocessed features (patch embeddings) from an encoder
  and outputs pixel-level features.
  """

  def __init__(self, config: VitDecoderConfig):
    super().__init__(config)

    if self.config.image_size % self.config.patch_size != 0:
      raise ValueError(
          f"Image size ({self.config.image_size}) must be divisible by patch"
          f" size ({self.config.patch_size})"
      )

    self.num_patches = (self.config.image_size // self.config.patch_size) ** 2

    # Project encoder features to decoder hidden size
    self.decoder_embed = nn.Linear(
        config.encoder_hidden_size, config.hidden_size, bias=True
    )

    # Positional embeddings for patch tokens
    if config.pos_emb_type == "trainable":
      self.decoder_pos_embed = nn.Parameter(
          torch.zeros(1, self.num_patches, config.hidden_size),
          requires_grad=True,
      )
      # Initialize with Gaussian with scale 0.02
      nn.init.normal_(self.decoder_pos_embed, std=0.02)
    elif config.pos_emb_type == "sincos2d":
      self.decoder_pos_embed = nn.Parameter(
          torch.zeros(1, self.num_patches, config.hidden_size),
          requires_grad=False,
      )
      # Initialize with 2D sin-cos embeddings
      grid_size = int(self.num_patches**0.5)
      sincos2d = positional_embeddings.Sincos2dEmbeddings(self.config)
      decoder_pos_embed = sincos2d.compute_sincos2d_embeddings(
          h=grid_size,
          w=grid_size,
          hidden_dim=self.decoder_pos_embed.shape[-1],
      )
      self.decoder_pos_embed.data.copy_(decoder_pos_embed)
    else:
      raise ValueError(f"Unsupported pos_emb_type: {config.pos_emb_type}")

    self.decoder_transformer = transformers.models.vit.modeling_vit.ViTEncoder(
        config
    )

    self.decoder_norm = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )
    self.decoder_pred = nn.Sequential(
        nn.Linear(
            config.hidden_size,
            config.patch_size**2 * config.output_dims,
            bias=True,
        ),
        nn.Tanh(),
    )

  def patch_to_image(self, patches: torch.Tensor) -> torch.Tensor:
    batch_size, num_patches, num_channels = patches.shape
    patch_size = self.config.patch_size
    output_dims = self.config.output_dims
    image_size = self.config.image_size

    assert num_channels == patch_size * patch_size * output_dims
    assert num_patches == self.num_patches

    patches_per_dim = int(num_patches**0.5)

    patches = patches.reshape(
        batch_size,
        patches_per_dim,
        patches_per_dim,
        patch_size,
        patch_size,
        output_dims,
    )

    patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
    return patches.reshape(batch_size, image_size, image_size, output_dims)

  def forward(self, hidden_states: torch.Tensor):
    """Forward pass of the decoder.

    Args:
      hidden_states: Preprocessed features (patch embeddings) from the
          encoder, of shape (batch_size, num_patches, encoder_hidden_size).

    Returns:
      The pixel-level features, of shape (batch_size, image_size, image_size,
      output_dims).
    """
    _, num_patches, _ = hidden_states.shape
    if num_patches != self.num_patches:
      raise ValueError(
          f"Input hidden_states have {num_patches} patches, but expected"
          f" {self.num_patches} for image size {self.config.image_size} and"
          f" patch size {self.config.patch_size}"
      )

    # Embed tokens to decoder hidden size
    # (batch_size, num_patches, decoder_hidden_size)
    features = self.decoder_embed(hidden_states)

    # Add positional embeddings
    features = features + self.decoder_pos_embed

    # Apply Transformer layers
    features = self.decoder_transformer(features).last_hidden_state

    # Apply layer norm
    features = self.decoder_norm(features)

    # Predictor projection
    logits = self.decoder_pred(features)  # (B, N, P*P*output_dims)

    # Reshape to image format
    # (batch_size, image_size, image_size, output_dims)
    return self.patch_to_image(logits)


class ViTEncoderDecoderModel(transformers.PreTrainedModel):
  """A ViT encoder-decoder model for dense prediction tasks."""

  config_class = EncoderDecoderConfig

  def __init__(self, config: EncoderDecoderConfig):
    super().__init__(config)

    if not config.encoder_config:
      raise ValueError("encoder_config must be provided.")
    if not config.decoder_config:
      raise ValueError("decoder_config must be provided.")

    self.encoder_config = config.encoder_config
    self.decoder_config = config.decoder_config

    assert (
        self.decoder_config.encoder_hidden_size
        == self.encoder_config.hidden_size
    )

    self.encoder = vits.PretrainedRemoteSensingVit(config.encoder_config)
    self.norm = nn.LayerNorm(
        self.encoder_config.hidden_size, eps=self.encoder_config.layer_norm_eps
    )
    self.decoder = ViTDecoder(config=self.decoder_config)

    self.post_init()

  def forward(
      self,
      pixel_values: torch.Tensor,
  ):
    encoder_outputs = self.encoder(
        pixel_values=pixel_values,
    )
    encoder_features = encoder_outputs.last_hidden_state
    encoder_features = self.norm(encoder_features)

    return self.decoder(encoder_features)


class ViTUNetSegmentationModel(transformers.PreTrainedModel):
  """ViT-based segmentation model with a UNet-style decoder.

  The model extracts intermediate features from selected encoder layers and
  uses them in a decoder that progressively upsamples the representation to
  produce a segmentation mask. Intermediate encoder features are captured
  with forward hooks, which allow the model to access layer outputs during
  the forward pass without modifying the encoder itself.

  PyTorch forward hooks:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook

  For a dense prediction task (e.g., segmentation),
  config.decoder_config.decoder_dims must be
  compatible with the encoder ViT's patch size to allow upsampling the
  representation.
  Specifically, the ViT encoder patch size must be a power of 2, and for patch
  size of 2^n the length of config.decoder_config.decoder_dims should be at
  least (n + 1) to produce output at the same resolution as the input image.
  If the length of config.decoder_config.decoder_dims is larger than n+1
  (for patch size 2^n) then the dense outputs will be a super-resolution of
  the input image.

  For example, for patch size 16 (n=4), if decoder_dims is of length 5, then
  the output segmentation map will be original resolution, and if decoder_dims
  is of length 6, then the output segmentation map will be at 2x
  super-resolution.
  """

  def __init__(
      self,
      config: EncoderDecoderConfig,
  ):
    """Initializes the ViTSegmentationModel.

    Args:
        config: The configuration for the ViT encoder and the UNet-style
          decoder.
    """
    super().__init__(config)

    if not config.encoder_config:
      raise ValueError("encoder_config must be provided.")
    if not config.decoder_config:
      raise ValueError("decoder_config must be provided.")

    self.encoder_config = config.encoder_config
    self.decoder_config: SkipUNetDecoderConfig = config.decoder_config

    if not config.decoder_config.skip_connections:
      raise ValueError("skip_connections list must not be empty.")

    self.encoder: transformers.ViTModel = vits.PretrainedRemoteSensingVit(
        config.encoder_config
    )
    self.decoder: architectures.ViTUNetDecoder = architectures.ViTUNetDecoder(
        encoder_dim=self.decoder_config.encoder_hidden_size,
        decoder_channels=self.decoder_config.decoder_dims,
        output_dims=self.decoder_config.output_dims,
    )
    self.patch_size: int = self.encoder_config.patch_size
    self._features: dict[str, torch.Tensor] = {}
    self.hook_layer_indices: tuple[int, ...] = (
        self.decoder_config.skip_connections
    )
    self.feat_names: list[str] = [f"feat{i}" for i in self.hook_layer_indices]
    self._hook_handles: list[Any] = []

    def get_activation(name: str) -> Callable[[nn.Module, Any, Any], None]:
      """Returns a hook function to capture activations."""

      def hook(module: nn.Module, inputs: Any, output: Any) -> None:
        del module, inputs
        if isinstance(output, tuple):
          output = output[0]
        self._features[name] = output

      return hook

    for i, name in zip(self.hook_layer_indices, self.feat_names):
      handle = self.encoder.encoder.layer[i].register_forward_hook(
          get_activation(name)
      )
      self._hook_handles.append(handle)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the segmentation model.

    Args:
        x: Input image tensor.

    Returns:
        Output segmentation map from the decoder.

    Raises:
        RuntimeError: If a feature expected from a hook is not found,
          indicating an issue with the hook setup or encoder structure.
    """
    self._features = {}

    _ = self.encoder(x)

    features = []

    b = x.shape[0]
    h = x.shape[2] // self.patch_size
    w = x.shape[3] // self.patch_size

    for name in self.feat_names:
      if name not in self._features:
        raise RuntimeError(
            f"Missing hooked feature '{name}'. Check hook_layer_indices and "
            "encoder structure."
        )

      feat = self._features[name]

      if feat.shape[1] == h * w + 1:
        feat = feat[:, 1:, :]

      if feat.shape[1] != h * w:
        raise ValueError(
            f"Feature '{name}' has incompatible token count {feat.shape[1]} "
            f"for spatial size ({h}, {w})."
        )

      feat = feat.transpose(1, 2).reshape(b, -1, h, w)
      features.append(feat)

    return self.decoder(features)
