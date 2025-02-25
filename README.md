# Multimodal Pragmatic Jailbreak on Text-to-image Models

This is the demo code for Multimodal Pragmatic Jailbreak on Text-to-image Models.   

<a href='https://huggingface.co/datasets/tongliuphysics/multimodalpragmatic'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MultimodalJailbreak-blue'></a> 

<div align="center">

<img src="files/111.png" width="99%">

</div>

<h2> Installation </h2>
To get started, install the package: 

```bash
git clone https://github.com/multimodalpragmatic/multimodalpragmatic.git
cd multimodalpragmatic
pip install -r requirements.txt
```

<h2> How to run </h2>  

For model {model_name} on {dataclass} category of MPUP dataset:   

```bash
python t2i.py -model_name ${model_name} -prompt_mode "sign" -dataclass ${dataclass}
```
  
E.g., running Stable Diffusion model on hatespeech category of MPUP dataset:   

```bash
python t2i.py -model_name "SD" -prompt_mode "sign" -dataclass "hatespeech"
```

For running the Glyphcontrol model on MPUP dataset, please first follow their instructions to download the checkpoint file, then run 
```bash
inference_glyphcontrol.py
```

<h2> How to evaluate multimodal pragmatic jailbreak </h2>  

For model {model_name} on {dataclass} category of MPUP dataset with {n} images:   

```bash
python multimodal_classification.py -model_name ${model_name} -prompt_mode "sign" -dataclass ${dataclass} -img_num ${n}
```

E.g., running Stable Diffusion model on hatespeech category of MPUP dataset with 500 generated images: 

```bash
python multimodal_classification.py -model_name "SD" -prompt_mode "sign" -dataclass "hatespeech" -img_num 500
```

<h2> How to evaluate visual text rendering </h2>  

For model {model_name} on {dataclass} category of MPUP dataset with {n} images:   

```bash
python ocr_evaluate.py -model_name ${model_name0} -prompt_mode "sign" -dataclass ${dataclass0}
```
E.g., running evaluation of visual text rendering on images generated by Stable Diffusion model on hatespeech category of MPUP dataset: 

```bash
python ocr_evaluate.py -model_name "SD" -prompt_mode "sign" -dataclass "hatespeech" 
```

<h2> How to cite this work </h2> 
```bash
@article{liu2024multimodal,
  title={Multimodal Pragmatic Jailbreak on Text-to-image Models},
  author={Liu, Tong and Lai, Zhixin and Zhang, Gengyuan and Torr, Philip and Demberg, Vera and Tresp, Volker and Gu, Jindong},
  booktitle={arXiv preprint arxiv:2409.19149},
  year={2024}
}
```
