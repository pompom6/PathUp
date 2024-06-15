import torch
from PIL import Image
from transformers import CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    ) 
import numpy as np
from torchvision import transforms
from modules import PWTT

VALID_PROMPTS = [
    '',
    ]

print('total images: ', str(len(VALID_PROMPTS)))

device = 'cuda:0'
base_model_path = ''

unet = UNet2DConditionModel.from_pretrained('', torch_dtype=torch.bfloat16)
text_encoder = CLIPTextModel.from_pretrained('', torch_dtype=torch.bfloat16)
vae = AutoencoderKL.from_pretrained('', torch_dtype=torch.bfloat16)

pipe = PWTT.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16,
    unet=unet, 
    text_encoder=text_encoder, 
    vae=vae, 
    
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# generate image
for p_prompt in VALID_PROMPTS:
    cond_img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ])
    cond_img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    cond_img = cond_img_transforms(Image.fromarray(cond_img)).unsqueeze(0)
    
    strength = 1.0
    gs = 6.5
    print(p_prompt)
    for i in range(1):
        image = pipe(
            p_prompt, 
            num_inference_steps=25, 
            image = cond_img,
            guidance_scale= gs,
            strength=0.8, 
            large_latent_height=1024*2,
            large_latent_width=1024*2,
            ).images[0]
        save_path = 'show.png'
        print(save_path)
        image.save(save_path)

