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

"""Earth AI Remote Sensing ViT models in PyTorch.

The implementation is based on the Huggingface Transformers library:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
"""

from typing import Optional, Union

from remote_sensing.models import positional_embeddings
import torch
from torch import nn
import transformers
from transformers import modeling_outputs
from transformers.models.vit import modeling_vit


class PretrainedRemoteSensingVit(transformers.ViTPreTrainedModel):
  """A Pre-trained ViT architecture for Earth AI Remote Sensing RGB imagery.

  Adapted from the implementation of modelings_vit.ViTModel with some changes:
    - Using sincos2d positional embeddings instead of trainable positional
      embeddings.
    - Making the final layer norm optional.
  """

  def __init__(
      self,
      config: transformers.ViTConfig,
      add_layer_norm: bool = False,
      add_pooling_layer: bool = False,
  ):
    super().__init__(config)
    self.config = config

    self.embeddings = positional_embeddings.Sincos2dEmbeddings(config)
    self.encoder = modeling_vit.ViTEncoder(config)

    if add_layer_norm:
      self.layernorm = nn.LayerNorm(
          config.hidden_size, eps=config.layer_norm_eps
      )
    else:
      self.layernorm = None

    self.pooler = transformers.ViTPooler(config) if add_pooling_layer else None

    # Initialize weights and apply final processing
    self.post_init()

  def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
    """Prunes heads of the model.

    Args:
      heads_to_prune: dict of {layer_num: list of heads to prune in this layer}

    See base class ViTPreTrainedModel.
    """
    for layer, heads in heads_to_prune.items():
      self.encoder.layer[layer].attention.prune_heads(heads)

  def forward(
      self,
      pixel_values: Optional[torch.Tensor] = None,
      head_mask: Optional[torch.Tensor] = None,
  ) -> Union[
      tuple[torch.Tensor, ...], modeling_outputs.BaseModelOutputWithPooling
  ]:

    if pixel_values is None:
      raise ValueError("You have to specify pixel_values")

    embedding_output = self.embeddings(pixel_values).float()

    encoder_outputs = self.encoder(embedding_output, head_mask=head_mask)
    sequence_output = encoder_outputs[0]

    if self.layernorm:
      sequence_output = self.layernorm(sequence_output)

    pooled_output = (
        self.pooler(sequence_output) if self.pooler is not None else None
    )

    if not self.config.use_return_dict:
      head_outputs = (
          (sequence_output, pooled_output)
          if pooled_output is not None
          else (sequence_output,)
      )
      return head_outputs + encoder_outputs[1:]

    return modeling_outputs.BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
