import io
import threading
import typing as T

import pydub
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import numpy as np

from riffusion.audio_splitter import AudioSplitter
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# TODO(hayk): Add URL params

DEFAULT_CHECKPOINT = "home/ubuntu/ai_directory/training/riffusion-model"

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


def run_img2img_magic_mix(
    prompt: str,
    init_image: Image.Image,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    kmin: float,
    kmax: float,
    mix_factor: float,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
):
    """
    Run the magic mix pipeline for img2img.
    """
    with pipeline_lock():
        pipeline = load_magic_mix_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        return pipeline(
            init_image,
            prompt=prompt,
            kmin=kmin,
            kmax=kmax,
            mix_factor=mix_factor,
            seed=seed,
            guidance_scale=guidance_scale,
            steps=num_inference_steps,
        )

def load_magic_mix_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
):
    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint,
        custom_pipeline="magic_mix",
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, pipeline.scheduler.config)

    return pipeline
    
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


def spectrogram_image_from_audio(
    segment: pydub.AudioSegment,
    params: SpectrogramParams,
    device: str = "cuda",
) -> Image.Image:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.spectrogram_image_from_audio(segment)

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


def slice_audio_into_clips(
    segment: pydub.AudioSegment, clip_start_times: T.Sequence[float], clip_duration_s: float, duration_s: float
) -> T.List[pydub.AudioSegment]:
    """
    Slice an audio segment into a list of clips of a given duration at the given start times.
    """
    clip_segments: T.List[pydub.AudioSegment] = []
    for i, clip_start_time_s in enumerate(clip_start_times):
        clip_start_time_ms = int(clip_start_time_s * 1000)
        clip_duration_ms = int(clip_duration_s * 1000)
        duration_ms = int(duration_s * 1000)
        print(clip_start_time_ms, clip_start_time_ms + clip_duration_ms, duration_ms)
        if(clip_start_time_ms + clip_duration_ms > duration_ms):
            clip_segment = segment[clip_start_time_ms : duration_ms]
        else:
            clip_segment = segment[clip_start_time_ms : clip_start_time_ms + clip_duration_ms]

        if i == len(clip_start_times) - 1 and clip_segment.duration_seconds < clip_duration_s:
            silence_duration = 0.5
            silence_segment = pydub.AudioSegment.silent(duration=int(silence_duration * 1000))
            clip_segment += silence_segment

        clip_segments.append(clip_segment)

    return clip_segments

def scale_image_to_32_stride(image: Image.Image) -> Image.Image:
    """
    Scale an image to a size that is a multiple of 32.
    """
    closest_width = int(np.ceil(image.width / 32) * 32)
    closest_height = int(np.ceil(image.height / 32) * 32)
    return image.resize((closest_width, closest_height), Image.BICUBIC)


def audio_segment_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)


def get_clip_params(advanced: bool = False) -> T.Dict[str, T.Any]:
    """
    Render the parameters of slicing audio into clips.
    """
    p: T.Dict[str, T.Any] = {}


    p["start_time_s"] = 0.0
    
    p["duration_s"] = 25.0

    p["clip_duration_s"] = 7.0

    p["overlap_duration_s"] = 0.4

    return p