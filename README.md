# IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models

<div align="center">

[**Project Page**](https://ip-adapter.github.io) **|** [**Paper (ArXiv)**](https://arxiv.org/abs/2308.06721)
</div>

---


## Introduction

we present IP-Adapter, an effective and lightweight
adapter to achieve image prompt capability for the pretrained
text-to-image diffusion models. An IP-Adapter
with only 22M parameters can achieve comparable or even
better performance to a fine-tuned image prompt model. IPAdapter
can be generalized not only to other custom models
fine-tuned from the same base model, but also to controllable
generation using existing controllable tools. Moreover, the image prompt
can also work well with the text prompt to accomplish multimodal
image generation.

![arch](assets/figs/fig1.png)

## Release
- [2023/8/16] 🔥 We release the code and models.


## Dependencies
- diffusers >= 0.19.3

## Download Models

you can download models from [here](https://huggingface.co/h94/IP-Adapter). To run the demo, you should also download the following models:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE)

## How to Use

- [**ip_adapter_demo**](ip_adapter_demo.ipynb): image variations, image-to-image, and inpainting with image prompt.

![image variations](assets/demo/image_variations.jpg)

![image-to-image](assets/demo/image-to-image.jpg)

![inpainting](assets/demo/inpainting.jpg)
- [**ip_adapter_controlnet_demo**](ip_adapter_controlnet_demo.ipynb): structural generation with image prompt.

![structural_cond](assets/demo/structural_cond.jpg)

- [**ip_adapter_multimodal_prompts_demo**](ip_adapter_multimodal_prompts_demo.ipynb): generation with multimodal prompts.

![multi_prompts](assets/demo/multi_prompts.jpg)


## Citation
If you find IP-Adapter useful for your your research and applications, please cite using this BibTeX:
```bibtex
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```
