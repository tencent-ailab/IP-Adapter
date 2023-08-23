# IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models

<div align="center">

[**Project Page**](https://ip-adapter.github.io) **|** [**Paper (ArXiv)**](https://arxiv.org/abs/2308.06721)
</div>

---


## Introduction

we present IP-Adapter, an effective and lightweight
adapter to achieve image prompt capability for the pre-trained
text-to-image diffusion models. An IP-Adapter
with only 22M parameters can achieve comparable or even
better performance to a fine-tuned image prompt model. IP-Adapter
can be generalized not only to other custom models
fine-tuned from the same base model, but also to controllable
generation using existing controllable tools. Moreover, the image prompt
can also work well with the text prompt to accomplish multimodal
image generation.

![arch](assets/figs/fig1.png)

## Release
- [2023/8/23] 🔥 Add code and models of IP-Adapter with fine-grained features. The demo is [here](ip_adapter-plus_demo.ipynb).
- [2023/8/18] 🔥 Add code and models for [SDXL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). The demo is [here](ip_adapter_sdxl_demo.ipynb).
- [2023/8/16] 🔥 We release the code and models.


## Dependencies
- diffusers >= 0.19.3

## Download Models

you can download models from [here](https://huggingface.co/h94/IP-Adapter). To run the demo, you should also download the following models:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE)
- [ControlNet models](https://huggingface.co/lllyasviel)

## How to Use

- [**ip_adapter_demo**](ip_adapter_demo.ipynb): image variations, image-to-image, and inpainting with image prompt.
- [![**ip_adapter_demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tencent-ailab/IP-Adapter/blob/main/ip_adapter_demo.ipynb) 

![image variations](assets/demo/image_variations.jpg)

![image-to-image](assets/demo/image-to-image.jpg)

![inpainting](assets/demo/inpainting.jpg)

- [**ip_adapter_controlnet_demo**](ip_adapter_controlnet_demo.ipynb): structural generation with image prompt.
- [![**ip_adapter_controlnet_demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tencent-ailab/IP-Adapter/blob/main/ip_adapter_controlnet_demo.ipynb) 

![structural_cond](assets/demo/structural_cond.jpg)

- [**ip_adapter_multimodal_prompts_demo**](ip_adapter_multimodal_prompts_demo.ipynb): generation with multimodal prompts.
- [![**ip_adapter_multimodal_prompts_demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tencent-ailab/IP-Adapter/blob/main/ip_adapter_multimodal_prompts_demo.ipynb) 

![multi_prompts](assets/demo/multi_prompts.jpg)

- [**ip_adapter-plus_demo**](ip_adapter-plus_demo.ipynb): the demo of IP-Adapter with fine-grained features.

![ip_adpter_plus_image_variations](assets/demo/ip_adpter_plus_image_variations.jpg )
![ip_adpter_plus_multi](assets/demo/ip_adpter_plus_multi.jpg )


**Best Practice**
- If you only use the image prompt, you can set the `scale=1.0` and `text_prompt=""`(or some generic text prompts, e.g. "best quality", you can also use any negative text prompt). If you lower the `scale`, more diverse images can be generated, but they may not be as consistent with the image prompt.
- For multimodal prompts, you can adjust the `scale` to get the best results. In most cases, setting `scale=0.5` can get good results. For the version of SD 1.5, we recommend using community models to generate good images.

## Citation
If you find IP-Adapter useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```
