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

"""Utility functions for Vertex AI remote sensing models."""

import base64
import dataclasses
import io

from google.cloud import aiplatform
from PIL import Image

SERVE_DOCKER_URI = "us-docker.pkg.dev/vertex-ai-restricted/vertex-vision-model-garden-dockers/remote-sensing-serve-tf-gpu:release"


@dataclasses.dataclass
class ModelConfig:
  """Model config for remote sensing models."""

  model_id: str
  model_name: str
  model_path: str


@dataclasses.dataclass
class PlatformConfig:
  """Platform config for remote sensing models."""

  machine_type: str
  accelerator_type: str | None
  accelerator_count: int | None


MODEL_CONFIGS = {
    "OWLVIT": ModelConfig(
        model_id="earth-ai-imagery-owlvit-eap-10-2025",
        model_name="publishers/google/models/remote_sensing_owlvit",
        model_path="gs://vertex-model-garden-remote-sensing-access/models/OVD_OWL-ViT_So400M_RGB1008_V1",
    ),
    "MAMMUT": ModelConfig(
        model_id="earth-ai-imagery-mammut-eap-10-2025",
        model_name="publishers/google/models/remote_sensing_mammut",
        model_path="gs://vertex-model-garden-remote-sensing-access/models/MaMMUT_So400M_RGB224_V1",
    ),
}


PLATFORM_CONFIGS = {
    "CPU": PlatformConfig(
        machine_type="e2-standard-8",
        accelerator_type=None,
        accelerator_count=None,
    ),
    "NVIDIA_L4": PlatformConfig(
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
    ),
    "NVIDIA_A100_80GB": PlatformConfig(
        machine_type="a2-ultragpu-1g",
        accelerator_type="NVIDIA_A100_80GB",
        accelerator_count=1,
    ),
}


def create_model(
    display_name: str,
    model_type: str,
    model_mode: str,
    accelerator: str,
    docker_uri: str = SERVE_DOCKER_URI,
    batch_model: bool = False,
) -> aiplatform.Model:
  """Creates a Remote Sensing model from Model Garden.

  Args:
    display_name: The display name of the model.
    model_type: The type of the model, e.g. MAMMUT, OWLVIT.
    model_mode: The mode of the model, e.g. IMAGE_ONLY, TEXT_ONLY, COMBINED.
    accelerator: The accelerator to use for the model.
    docker_uri: The docker URI to use for the model.
    batch_model: Whether the model is a batch model.

  Returns:
    The model.
  """
  if model_type not in MODEL_CONFIGS:
    raise ValueError(f"Model type is not supported {model_type}")

  model_config = MODEL_CONFIGS[model_type]

  env_vars = {
      "DEPLOY_SOURCE": "notebook",
      "MODEL_ID": model_config.model_id,
      "MODEL_PATH": model_config.model_path,
      "MODEL_TYPE": model_type,
      "MODEL_MODE": model_mode,
      "PLATFORM": "CPU" if accelerator == "CPU" else "GPU",
  }

  model = aiplatform.Model.upload(
      display_name=display_name,
      serving_container_image_uri=docker_uri,
      serving_container_ports=[8080],
      serving_container_predict_route="/predict",
      serving_container_health_route="/health",
      serving_container_environment_variables=env_vars,
      # Remove the model garden source model name for batch models.
      model_garden_source_model_name=None
      if batch_model
      else model_config.model_name,
  )
  return model


def deploy_model(
    endpoint_name: str,
    model: aiplatform.Model,
    accelerator: str,
    service_account: str,
    min_replica_count: int,
    max_replica_count: int,
    use_dedicated_endpoint: bool = True,
) -> aiplatform.Endpoint:
  """Deploys a model to an endpoint."""

  endpoint = aiplatform.Endpoint.create(
      endpoint_name, dedicated_endpoint_enabled=use_dedicated_endpoint
  )

  if accelerator not in PLATFORM_CONFIGS:
    raise ValueError(f"Accelerator config is not supported {accelerator}")

  platform_config = PLATFORM_CONFIGS[accelerator]

  model.deploy(
      endpoint=endpoint,
      machine_type=platform_config.machine_type,
      accelerator_type=platform_config.accelerator_type,
      accelerator_count=platform_config.accelerator_count,
      service_account=service_account,
      deploy_request_timeout=1800,
      enable_access_logging=True,
      min_replica_count=min_replica_count,
      max_replica_count=max_replica_count,
      sync=True,
      system_labels={
          "NOTEBOOK_NAME": "model_garden_remote_sensing_deployment.ipynb"
      },
  )
  return endpoint


def png_bytes(image: Image.Image) -> bytes:
  """Encodes the `image` as PNG bytes."""
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  return buf.getvalue()


def b64_png(image: Image.Image) -> str:
  """Converts the `image` to a b64 encoded PNG bytes."""
  return base64.b64encode(png_bytes(image)).decode()
