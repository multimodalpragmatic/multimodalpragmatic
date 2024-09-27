import os
import json
import torch
import random
import requests
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import StableDiffusionPipelineSafe, DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline, KDPM2AncestralDiscreteScheduler
from diffusers import AutoPipelineForText2Image, AutoencoderKL, DiffusionPipeline
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
from datasets import load_dataset





def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def main(args):

    dataclass = args.dataclass
    dataset = load_dataset("tongliuphysics/multimodalpragmatic")["train"]
    data = dataset.filter(lambda x: x["class"] == dataclass)
    print("There are totally {} prompts".format(len(data)))




    # safe SD
    if args.model_name == "SLD":
        pipe = StableDiffusionPipelineSafe.from_pretrained(
            "AIML-TUDA/stable-diffusion-safe", torch_dtype=torch.float16
        ).to("cuda")
    elif args.model_name == "SD":
        repo_id = "stabilityai/stable-diffusion-2-base"
        pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
    elif args.model_name == "SDXL":
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.to("cuda")
    elif args.model_name == "deepfloyd":
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        stage_1.to("cuda")

        stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        stage_2.to("cuda")

        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker,
                          "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules,
                                                    torch_dtype=torch.float16)
        stage_3.to("cuda")

    elif args.model_name == "Proteus":
        # Load VAE component
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )

        # Configure the pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "dataautogpt3/ProteusV0.3",
            vae=vae,
            torch_dtype=torch.float16
        )
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')

    elif args.model_name == "OpenDalle":
        pipe = AutoPipelineForText2Image.from_pretrained('dataautogpt3/OpenDalleV1.1',
                                                             torch_dtype=torch.float16).to('cuda')
        pipe.to('cuda')

    elif args.model_name == "pixartalpha":
        pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)
        pipe.to('cuda')




    base_res_path = "./results/{}".format(args.model_name)
    if not os.path.exists(base_res_path):
        # Create the directory
        os.makedirs(base_res_path)

    succ_gen = 0
    unsucc_list = []
    for i, _data in tqdm(enumerate(data)):
        if args.prompt_mode == "ins":

            _prompt = "Generate an image of '" + _data["image_prompt"] + "'" \
                      + ", then add text clearly on this image (please make sure the text is accurate), " \
                      + "'" + _data["text_prompt"] + "'"

        elif args.prompt_mode == "sign":
            _prompt = _data["image_prompt"] + ", with a sign that says, '" + _data["text_prompt"] + "'"

        elif args.prompt_mode == "text":
            _prompt = _data["image_prompt"] + ", with the text'" + _data["text_prompt"] + "'"

        elif args.prompt_mode == "image_only":
            _prompt = _data["image_prompt"]

        


        try:
            if args.model_name == "SD":
                images = pipe(prompt=_prompt, num_inference_steps=50).images[0]

            elif args.model_name == "SDXL":
                images = pipe(prompt=_prompt).images[0]

            elif args.model_name == "SLD":
                images = pipe(prompt=_prompt, **SafetyConfig.MEDIUM).images[0]

            elif args.model_name == "deepfloyd":
                prompt_embeds, negative_embeds = stage_1.encode_prompt(_prompt)
                generator = torch.manual_seed(0)
                image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                                generator=generator,
                                output_type="pt").images
                image = stage_2(
                    image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                    generator=generator, output_type="pt"
                ).images
                images = stage_3(prompt=_prompt, image=image, generator=generator, noise_level=100).images[0]

            elif args.model_name == "Proteus":
                # Define prompts and generate image
                negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"
                images = pipe(
                    _prompt,
                    negative_prompt=negative_prompt,
                    width=1024,
                    height=1024,
                    guidance_scale=7,
                    num_inference_steps=20
                ).images[0]

            elif args.model_name == "OpenDalle":
                images = pipe(_prompt).images[0]

            elif args.model_name == "pixartalpha":
                images = pipe(_prompt).images[0]

            directory_path = "./results/{}_{}_{}".format(args.model_name, args.prompt_mode, dataclass)

            # Check if the directory exists, if not create it
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            image_path = "./results/{}_{}_{}/{}.png".format(args.model_name, args.prompt_mode, dataclass, i)
            images.save(image_path)
            succ_gen += 1

        except:
            unsucc_list.append(i)

    print("Successfully generate rate: ", succ_gen / len(data))

    print("Unpassed prompts: ", unsucc_list)

    # save unsucc_list
    txt_path = "./results/unsucc_list/{}_{}_{}.txt".format(args.model_name, args.prompt_mode, dataclass)
    with open(txt_path, 'w') as file:
        for i in unsucc_list:
            file.write(str(i)+" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, default="SD")
    parser.add_argument("-prompt_mode", type=str, default="sign")
    parser.add_argument("-dataclass", type=str, default="hatespeech")

    args = parser.parse_args()

    set_seed(4242424242)

    main(args)















