import io
import threading
import typing as T

import pydub
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from riffusion.audio_splitter import AudioSplitter
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# TODO(hayk): Add URL params

DEFAULT_CHECKPOINT = "riffusion/riffusion-model-v1"

AUDIO_EXTENSIONS = ["mp3", "wav", "flac", "webm", "m4a", "ogg"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]

SCHEDULER_OPTIONS = [
    "DPMSolverMultistepScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
]

def run_img2img(
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: T.Optional[str] = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
    progress_callback: T.Optional[T.Callable[[float], T.Any]] = None,
) -> Image.Image:
    with pipeline_lock():
        pipeline = load_stable_diffusion_img2img_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)

        def callback(step: int, tensor: torch.Tensor, foo: T.Any) -> None:
            if progress_callback is not None:
                progress_callback(step / num_expected_steps)

        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt or None,
            num_images_per_prompt=1,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )

        return result.images[0]
    
def pipeline_lock() -> threading.Lock:
    """
    Singleton lock used to prevent concurrent access to any model pipeline.
    """
    return threading.Lock()

def load_stable_diffusion_img2img_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionImg2ImgPipeline:
    """
    Load the image to image pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)

    return pipeline

def spectrogram_image_converter(
    params: SpectrogramParams,
    device: str = "cuda",
) -> SpectrogramImageConverter:
    return SpectrogramImageConverter(params=params, device=device)


def get_scheduler(scheduler: str, config: T.Any) -> T.Any:
    """
    Construct a denoising scheduler from a string.
    """
    if scheduler == "PNDMScheduler":
        from diffusers import PNDMScheduler

        return PNDMScheduler.from_config(config)
    elif scheduler == "DPMSolverMultistepScheduler":
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(config)
    elif scheduler == "DDIMScheduler":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(config)
    elif scheduler == "LMSDiscreteScheduler":
        from diffusers import LMSDiscreteScheduler

        return LMSDiscreteScheduler.from_config(config)
    elif scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(config)
    elif scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler

        return EulerAncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")
    
def audio_segment_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)