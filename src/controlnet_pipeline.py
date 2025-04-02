import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from PIL import Image


def load_controlnet_pipeline(controlnet_ckpt, sd_ckpt):

    controlnet = ControlNetModel.from_pretrained(controlnet_ckpt, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_ckpt,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    return pipe


def run_inference(
        pipe: StableDiffusionControlNetPipeline,
        prompt: str,
        control_image: Image.Image,
        num_steps: int = 30,
        seed: int = 0
):
    generator = torch.manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        image=control_image
    ).images[0]

    return result
