from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
import copy
from typing import Optional, Union, List, Callable, Dict, Any
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import copy
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import numpy as np
from tqdm import tqdm
import copy

class randomTailor():
    def __init__(self, init_large, sample_per_tile=4, min_shift_s=None, device=None) -> None:
        # self.large_latent_width = large_latent_width #[*,*,256,256]
        if device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.lls = list(init_large.shape[:])
        self.large_latent_width = init_large.shape[-1]
        self.num_tile_w = self.large_latent_width//64
        self.sample_per_tile = 4
        self.tile_hw = self.large_latent_width//self.num_tile_w
        if min_shift_s == None:
            self.min_shift = self.large_latent_width/self.num_tile_w/sample_per_tile
        else:
            self.min_shift = min_shift_s
        # initialize small latent
        self.weight_mask = self.init_weight_mask()
        self.sample_points = []
        tile_all_sample_points = np.array([np.meshgrid(np.linspace(0, sample_per_tile-1, sample_per_tile), np.linspace(0, sample_per_tile-1, sample_per_tile))], dtype=np.uint8)
        self.tile_all_sample_points = tile_all_sample_points.reshape(2, -1).T.tolist()
        self.tile_all_sample_points.remove([0,0])
        self.chosen_samples = self.generate_sample_points()
        self.chosen_samples_control = self.generate_control_tiles_cor()
        self.large_latent = init_large

    def init_weight_mask(self):
        msk_tensor = torch.zeros((self.tile_hw, self.tile_hw)) 
        for i in range(-self.tile_hw//2,self.tile_hw//2):
            for j in range(-self.tile_hw//2,self.tile_hw//2):
                # if i ==-self.tile_hw//2-1 and j == 0:
                #     print(min(abs(i-0),abs(j-0)))
                msk_tensor[j,i]=min(abs(i-0),abs(j-0))
        msk_tensor = (1-0.5)*(msk_tensor-torch.min(msk_tensor))/(torch.max(msk_tensor)-torch.min(msk_tensor)) + 0.5
        return msk_tensor.to(torch.device(self.device))
            
    def tail_results(self, samples, out_mask=False):
        result_large = torch.zeros(self.lls)
        weight_mask = torch.ones_like(samples[0])*self.weight_mask  # spread weight mask to all dims 
        result_large_weight = torch.zeros(self.lls)
        sample_w = samples[0].shape[-1]
        for n, ((i,j), s) in enumerate(zip(self.chosen_samples, samples)):
            temp_weight = torch.zeros_like(result_large_weight)
            temp_val = torch.zeros_like(result_large)
            temp_weight[:,:,i:i+sample_w,j:j+sample_w] = weight_mask
            temp_val[:,:,i:i+sample_w,j:j+sample_w] = s*weight_mask
            result_large_weight += temp_weight 
            result_large += temp_val
        result_large = torch.div(result_large, result_large_weight)
        
        if out_mask:
            out_weight_mask = result_large_weight
            out_weight_mask[out_weight_mask>0]=1
            out_weight_mask = 1 - out_weight_mask
            
            return result_large.to(self.device).type(torch.bfloat16), out_weight_mask.to(self.device).type(torch.bfloat16)
        else:
            return result_large.to(self.device).type(torch.bfloat16)
            
    def generate_sample_points(self): 
        grids = []
        for ti in range(self.num_tile_w):
            for tj in range(self.num_tile_w):
                if ti == self.num_tile_w-1 or tj == self.num_tile_w-1:
                    grids.append((ti*self.tile_hw,tj*self.tile_hw))
                    if ti !=  self.num_tile_w-1:
                        grids.append((int(ti*self.tile_hw + 1/2*self.tile_hw), tj*self.tile_hw))
                    if tj !=  self.num_tile_w-1:
                        grids.append((ti*self.tile_hw, int(tj*self.tile_hw + 1/2*self.tile_hw)))
                else:
                    grids.append((ti*self.tile_hw,tj*self.tile_hw))
                    grids += [(int(ti*self.tile_hw + rr[0]*self.min_shift), 
                               int(tj*self.tile_hw + rr[1]*self.min_shift))
                              for rr in range(self.tile_all_sample_points, self.sample_per_tile-1)]
        return grids
    
    def generate_control_tiles_cor(self, control_img_width=None):
        if control_img_width is None:
            control_img_width = self.large_latent_width*8
        return tuple([(int(ii*control_img_width/self.large_latent_width), int(jj*control_img_width/self.large_latent_width)) 
                for ii, jj in self.chosen_samples])
    

class PWTT(StableDiffusionImg2ImgPipeline):
    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image = None,
        large_latent_height:int = None,
        large_latent_width:int = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess image
        # image = image.resize((large_latent_height,large_latent_width))
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        large_latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
            generator=generator,
        )
        
        tailor = randomTailor(large_latents, device='cuda')
        schedulers = [copy.deepcopy(self.scheduler) for i in tailor.chosen_samples]

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                all_tile_latents=[]
                pbar = tqdm(range(len(tailor.chosen_samples))) 
                for (ii,jj), sch, pi, in zip(tailor.chosen_samples, schedulers, pbar, ):
                    pbar.set_description('Processing {}/{}'.format(pi+1, len(tailor.chosen_samples)))
                    latents = large_latents[:,:,ii:ii+tailor.tile_hw, jj:jj+tailor.tile_hw]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    xx = sch.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    all_tile_latents.append(xx)
                    # print(xx.shape)
                    
                large_latents = tailor.tail_results(all_tile_latents)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, large_latents)

        if not output_type == "latent":
            image = self.vae.decode(large_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = large_latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__ == '__main__':
    xx = randomTailor(256,)
    print(xx.chosen_samples)
    print(len(xx.chosen_samples))
    # print(xx.weight_mask)
    tx = torch.ones([2,2,256,256])
    ti = [torch.ones(2,2,64,64) for i in range(len(xx.chosen_samples))]
    xx.tail_results(tx, ti)
    print('done')
