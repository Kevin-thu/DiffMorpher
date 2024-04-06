<p align="center">
  <h1 align="center">DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing</h1>
  <h3 align="center">CVPR 2024</h3>
  <p align="center">
    <a href="https://kevin-thu.github.io/homepage/"><strong>Kaiwen Zhang</strong></a>
    &nbsp;&nbsp;
    <a href="https://zhouyifan.net/about/"><strong>Yifan Zhou</strong></a>
    &nbsp;&nbsp;
    <a href="https://sheldontsui.github.io/"><strong>Xudong Xu</strong></a>
    &nbsp;&nbsp;
    <a href="https://xingangpan.github.io/"><strong>Xingang Pan<sep>✉</sep></strong></a>
    &nbsp;&nbsp;
    <a href="http://daibo.info/"><strong>Bo Dai</strong></a>
  </p>
  <br>

  <p align="center">
    <sep>✉</sep>Corresponding Author
  </p>

  <div align="center">
        <img src="./assets/teaser.gif", width="500">
  </div>

  <p align="center">
    <a href="https://arxiv.org/abs/2312.07409"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2312.07409-b31b1b.svg"></a>
    <a href="https://kevin-thu.github.io/DiffMorpher_page/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://twitter.com/sze68zkw"><img alt='Twitter' src="https://img.shields.io/twitter/follow/sze68zkw?label=%40KaiwenZhang"></a>
    <a href="https://twitter.com/XingangP"><img alt='Twitter' src="https://img.shields.io/twitter/follow/XingangP?label=%40XingangPan"></a>
  </p>
  <br>
</p>

## Web Demos

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/KaiwenZhang/DiffMorpher)

<p align="left">
  <a href="https://huggingface.co/spaces/Kevin-thu/DiffMorpher"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DiffMorpher-orange"></a>
</p>

<!-- Great thanks to [OpenXLab](https://openxlab.org.cn/home) for the NVIDIA A100 GPU support! -->

## Requirements
To install the requirements, run the following in your environment first:
```bash
pip install -r requirements.txt
```
To run the code with CUDA properly, you can comment out `torch` and `torchvision` in `requirement.txt`, and install the appropriate version of `torch` and `torchvision` according to the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

You can also download the pretrained model *Stable Diffusion v2.1-base* from [Huggingface](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), and specify the `model_path` to your local directory.

## Run Gradio UI
To start the Gradio UI of DiffMorpher, run the following in your environment:
```bash
python app.py
```
Then, by default, you can access the UI at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Run the code
You can also run the code with the following command:
```bash
python main.py \
  --image_path_0 [image_path_0] --image_path_1 [image_path_1] \ 
  --prompt_0 [prompt_0] --prompt_1 [prompt_1] \
  --output_path [output_path] \
  --use_adain --use_reschedule --save_inter
```
The script also supports the following options:

- `--image_path_0`: Path of the first image (default: "")
- `--prompt_0`: Prompt of the first image (default: "")
- `--image_path_1`: Path of the second image (default: "")
- `--prompt_1`: Prompt of the second image (default: "")
- `--model_path`: Pretrained model path (default: "stabilityai/stable-diffusion-2-1-base")
- `--output_path`: Path of the output image (default: "")
- `--save_lora_dir`: Path of the output lora directory (default: "./lora")
- `--load_lora_path_0`: Path of the lora directory of the first image (default: "")
- `--load_lora_path_1`: Path of the lora directory of the second image (default: "")
- `--use_adain`: Use AdaIN (default: False)
- `--use_reschedule`: Use reschedule sampling (default: False)
- `--lamb`: Hyperparameter $\lambda \in [0,1]$ for self-attention replacement, where a larger $\lambda$ indicates more replacements (default: 0.6)
- `--fix_lora_value`: Fix lora value (default: LoRA Interpolation, not fixed)
- `--save_inter`: Save intermediate results (default: False)
- `--num_frames`: Number of frames to generate (default: 50)
- `--duration`: Duration of each frame (default: 50)

Examples:
```bash
python main.py \
  --image_path_0 ./assets/Trump.jpg --image_path_1 ./assets/Biden.jpg \ 
  --prompt_0 "A photo of an American man" --prompt_1 "A photo of an American man" \
  --output_path "./results/Trump_Biden" \
  --use_adain --use_reschedule --save_inter
```

```bash
python main.py \
  --image_path_0 ./assets/vangogh.jpg --image_path_1 ./assets/pearlgirl.jpg \ 
  --prompt_0 "An oil painting of a man" --prompt_1 "An oil painting of a woman" \
  --output_path "./results/vangogh_pearlgirl" \
  --use_adain --use_reschedule --save_inter
```

```bash
python main.py \
  --image_path_0 ./assets/lion.png --image_path_1 ./assets/tiger.png \ 
  --prompt_0 "A photo of a lion" --prompt_1 "A photo of a tiger" \
  --output_path "./results/lion_tiger" \
  --use_adain --use_reschedule --save_inter
```

## License
The code related to the DiffMorpher algorithm is licensed under [LICENSE](LICENSE.txt). 

However, this project is mostly built on the open-sourse library [diffusers](https://github.com/huggingface/diffusers), which is under a separate license terms [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE). (Cheers to the community as well!)

## Citation

```bibtex
@article{zhang2023diffmorpher,
    title={DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing},
    author={Zhang, Kaiwen and Zhou, Yifan and Xu, Xudong and Pan, Xingang and Dai, Bo},
    journal={arXiv preprint arXiv:2312.07409},
    year={2023}
}
```
