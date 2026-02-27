# Earth AI Remote Sensing Models

*This is not an officially supported Google product.*

This project is not eligible for the [Google Open Source Software Vulnerability
Rewards Program](https://bughunters.google.com/open-source-security).

Visit the [Earth AI website](https://ai.google/earth-ai/) and
check the [technical report](https://arxiv.org/abs/2510.18318) for additional
information about Earth AI and the different model families.

## Research Papers

* A Recipe for Improving Remote Sensing VLM Zero Shot Generalization ([abstract](https://arxiv.org/abs/2503.08722), [pdf](https://arxiv.org/pdf/2503.08722))
* On-the-Fly OVD Adaptation with FLAME: Few-shot Localization via Active Marginal-Samples Exploration ([abstract](https://arxiv.org/pdf/2510.17670), [pdf](https://arxiv.org/pdf/2510.17670))
* Zero-Shot Multi-Spectral Learning: Reimagining a Generalist Multimodal Gemini 2.5 Model for Remote Sensing Applications ([abstract](https://arxiv.org/abs/2509.19087), [pdf](https://arxiv.org/pdf/2509.19087), [blog](https://developers.googleblog.com/unlocking-multi-spectral-data-with-gemini/), [colab](https://github.com/google-gemini/cookbook/blob/main/examples/multi_spectral_remote_sensing.ipynb))
* Enhancing Remote Sensing Representations Through Mixed-Modality Masked Autoencoding ([abstract](https://ieeexplore.ieee.org/document/10972577), [pdf](https://openaccess.thecvf.com/content/WACV2025W/GeoCV/papers/Linial_Enhancing_Remote_Sensing_Representations_Through_Mixed-Modality_Masked_Autoencoding_WACVW_2025_paper.pdf))

## Remote Sensing Models on Vertex AI Model Garden

[Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden/google/earth-ai/)
hosts the Classification and Retrieval Model for Remote Sensing
(the [MaMMUT embeddings model](https://arxiv.org/abs/2503.08722))
and the Object Detection for Remote Sensing (based on the OWL-ViT architecture).
Deployment instructions will be provided when access is granted.

Additional useful notebooks:

* [VLM Batch prediction](remote_sensing/vertex_ai/notebooks/vlm_batch_prediction_example.ipynb)

