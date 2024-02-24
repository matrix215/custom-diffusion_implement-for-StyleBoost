import img_rename2, os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import DiffusionPipeline

import gc, subprocess
from random import randrange
import argparse

def setting(model_dic,  gpu = 1):
    #GPU setting
    #torch.cuda.set_device(int(gpu))
    pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.unet.load_attn_procs("/home/kkko2/custom/jouner-ani-1000", weight_name="pytorch_custom_diffusion_weights.bin")
    pipe.load_textual_inversion("/home/kkko2/custom/jouner-ani-1000", weight_name="<new1>.bin")
    pipe.load_textual_inversion("/home/kkko2/custom/jouner-ani-1000", weight_name="<new2>.bin")
    
    return pipe
def txt2img(model_dic, pipe, prompt, negative_prompt, prompt_id, index):
   #height, width
    gc.collect()
    torch.cuda.empty_cache()
    
    #seed
    seed = randrange(300000000)
    
    print('user prompt : ',prompt)
    print('prompt id : ',prompt_id)
   
    num_samples = 5
    guidance_scale = 8
    num_inference_steps = 30
    height = 512 
    width = 512 

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
        ).images
    
    
    file_path = img_rename2.make_dic(model_dic, prompt_id)
    
    for img in images:
        img.save(file_path + f'/{index}.{seed}.png','PNG')
        index = index + 1
def main_process(prompt, negative_prompt, model_dic, prompt_id, index, gpu):
    
    pipe = setting(model_dic, gpu)
    txt2img(model_dic,pipe, prompt, negative_prompt, prompt_id, index)
    
if __name__=='__main__':
    os.system('mkdir -p ~/.huggingface')
    HUGGINGFACE_TOKEN = "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"
    os.system(f'echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token')
    
    
    prompt_id = dict()
    with open('prompt.txt','r') as f:
        prompt_lst = f.readlines()  
    
        for i in range(315,316): #0-750,750-마지막*3
    
                prompt_id[str(i+1)] = prompt_lst[i].rstrip('\n')
    negative_prompt = '(worst quality, low quality:1.2), canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), 3d render'
    model_dic = "/home/kkko2/custom/jouner-ani-1000" # 750학습 8장 미드 back 사진임
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type= int, default= 1)  
    args = parser.parse_args()
    
    for i in range(args.iter):
        index = 0 
        gpu = 2
        for prmpt_id in prompt_id.keys():
            prompt_id[prmpt_id] = '<new1> style, <new2> style, '+prompt_id[prmpt_id]
            main_process(prompt_id[prmpt_id], negative_prompt, model_dic, prmpt_id, index, gpu)