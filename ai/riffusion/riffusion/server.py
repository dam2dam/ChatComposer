"""
Flask server that serves the riffusion model as an API.
"""

import dataclasses
import io
import json
import logging
import time
import typing as T
from pathlib import Path
import sys

import dacite
import flask
import PIL
from flask_cors import CORS

import requests 
import wave
import pydub
from PIL import Image
import numpy as np
from random import randint
from riffusion.util import server_util

from riffusion.datatypes import InferenceInput, InferenceOutput, RiffusionInput, RiffusionOutput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import audio_util
from riffusion.util import base64_util
import json
import base64

# Flask app with CORS
app = flask.Flask(__name__)
CORS(app)

# Create a logger and set the desired log level
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler to write logs to the server.log file
file_handler = logging.FileHandler("server.log")
file_handler.setLevel(logging.INFO)

# Create a stream handler to write logs to the standard output (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set the formatter for the handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Global variable for the model pipeline
PIPELINE: T.Optional[RiffusionPipeline] = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")


def run_app(
    *,
    checkpoint="home/ubuntu/ai_directory/training/riffusion-modell",
    no_traced_unet: bool = False,
    device: str = "cuda",
    host: str = "0.0.0.0",
    port: int = 3313,
    debug: bool = False,
    ssl_certificate: T.Optional[str] = None,
    ssl_key: T.Optional[str] = None,
):
    """
    Run a flask API that serves the given riffusion model checkpoint.
    """
    # Initialize the model
    global PIPELINE
#     PIPELINE = RiffusionPipeline.load_checkpoint(
#         checkpoint=checkpoint,
#         use_traced_unet=not no_traced_unet,
#         device=device,
#     )

    args = dict(
        debug=False,
        threaded=False,
        host=host,
        port=port,
    )

    if ssl_certificate:
        assert ssl_key is not None
        args["ssl_context"] = (ssl_certificate, ssl_key)

    app.run(**args)  # type: ignore


@app.route("/")
def hello():
    return "hello"

@app.route("/riff", methods=["POST"])
def riff():
    """
    1. download wav file from source [X]
    2. wav file to spectrogram [X]
    3. spectogram to audio(mp3) [X]
    4. send audio [x]
    """
    start_time = time.time()

    # Parse the payload as JSON
    json_data = json.loads(flask.request.data)

    # Log the request
    logging.info(json_data)

    # Parse an InferenceInput dataclass from the payload
    try:
        inputs = dacite.from_dict(RiffusionInput, json_data)
    except dacite.exceptions.WrongTypeError as exception:
        logging.info(json_data)
        return str(exception), 400
    except dacite.exceptions.MissingValueError as exception:
        logging.info(json_data)
        return str(exception), 400   
     

    response = compute_riff_request(
        inputs=inputs,
        pipeline=PIPELINE,
    )

    return response


def compute_riff_request(
    inputs: RiffusionInput,
    pipeline: RiffusionPipeline,
)-> str:
    segment = download_wav(inputs.musicSource)
    logging.info("wav file downloaded!")

    # variables from audio_to_audio.py
    start_time_s = 0.0
    clip_duration_s = 20.0
    overlap_duration_s = 0.5
    batches = 1

    logging.info(segment.duration_seconds)
    
#     duration_s = min(clip_duration_s, segment.duration_seconds - start_time_s)
#     increment_s = clip_duration_s - overlap_duration_s
#     clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)    # spectorgram 

    # prompt_input_a = PromptInput(
    #     prompt=prompt,
    #     seed=seed,
    #     guidance=guidance,
    # )

    # use magic_mix
    prompt = inputs.prompt
    seed = randint(10, 100000)      

    magic_mix_kmin = 0.3
    magic_mix_kmax = 0.9
    magic_mix_mix_factor = 0.5
    
    clip_p = server_util.get_clip_params()
    start_time_s = clip_p["start_time_s"]
    clip_duration_s = clip_p["clip_duration_s"]
    overlap_duration_s = clip_p["overlap_duration_s"]

    duration_s = min(clip_p["duration_s"], segment.duration_seconds - start_time_s)
    increment_s = clip_duration_s - overlap_duration_s
    clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)

    # Add the last few seconds to the clip_start_times
    last_clip_start_time = clip_start_times[-1] + increment_s
    clip_start_times = np.append(clip_start_times, last_clip_start_time)
    
#     clip_p = server_util.get_clip_params()
#     start_time_s = clip_p["start_time_s"] # 0
#     clip_duration_s = clip_p["clip_duration_s"] # 5
#     overlap_duration_s = clip_p["overlap_duration_s"] # 0

#     duration_s = min(clip_p["duration_s"], segment.duration_seconds - start_time_s) # (20, 8 - 0) => 8
#     increment_s = clip_duration_s - overlap_duration_s # 5 - 0
#     clip_start_times = start_time_s + np.arange(0, duration_s - 3, increment_s) # (0, 5)
    
    print(clip_start_times)
    
    clip_segments = server_util.slice_audio_into_clips(
        segment=segment,
        clip_start_times=clip_start_times,
        clip_duration_s=clip_duration_s,
        duration_s = duration_s
    )
    logging.info("wav file sliced!")
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
        stereo=False,
    )

    # inference params
    denoising_strength = 0.7
    guidance_scale = 7.0
    num_inference_steps = 50
    scheduler="DPMSolverMultistepScheduler"
    device="cuda"
    # 모델 경로 잡아야 됨
    checkpoint="../training/riffusion-model"

    result_images: T.List[Image.Image] = []
    result_segments: T.List[pydub.AudioSegment] = []
    logging.info(f"number of clips : {len(clip_segments)}")
    
    for i, clip_segment in enumerate(clip_segments):
        audio_bytes = io.BytesIO()
        clip_segment.export(audio_bytes, format="wav")
        init_image: Image.Image = server_util.spectrogram_image_from_audio(
            clip_segment,
            params=params,
            device="cuda",
        )

        init_image_resized = server_util.scale_image_to_32_stride(init_image)
        image = server_util.run_img2img_magic_mix(
            prompt=prompt,
            init_image=init_image_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            kmin=magic_mix_kmin,
            kmax=magic_mix_kmax,
            mix_factor=magic_mix_mix_factor,
            device=device,
            scheduler=scheduler,
            checkpoint=checkpoint,
        )

        # Resize back to original size
        image = image.resize(init_image.size, Image.BICUBIC)

        result_images.append(image)

        riffed_segment = server_util.audio_segment_from_spectrogram_image(
            image=image,
            params=params,
            device=device,
        )
        result_segments.append(riffed_segment)

        audio_bytes = io.BytesIO()
        riffed_segment.export(audio_bytes, format="wav")
    logging.info(f"created segments: {len(result_segments)}")
    # Combine clips with a crossfade based on overlap
    combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=0.0)
    
    audio_bytes = io.BytesIO()
    combined_segment.export(audio_bytes, format="mp3")
    audio_bytes.seek(0)  # Move the file pointer to the beginning of the file

    # Assemble the output dataclass
    output = RiffusionOutput(
        audio=base64_util.encode(audio_bytes)
    )
    return base64_util.encode(audio_bytes)

def download_wav(url: str):
    # fetch the file contents using the requests library
    response = requests.get(url)
    wav_data = response.content

    # Convert the byte array to AudioSegment
    audio_segment = pydub.AudioSegment.from_wav(io.BytesIO(wav_data))

    return audio_segment

def download_local_wav():
    # specify the path to the WAV file
    wav_path = '/path/to/file.wav'

    # open the WAV file using the wave library
    with wave.open(wav_path, 'rb') as wav_file:
        # read the WAV file data into a buffer
        data = wav_file.readframes(wav_file.getnframes())

    return data

@app.route("/run_inference/", methods=["POST"])
def run_inference():
    """
    Execute the riffusion model as an API.

    Inputs:
        Serialized JSON of the InferenceInput dataclass

    Returns:
        Serialized JSON of the InferenceOutput dataclass
    """
    start_time = time.time()

    # Parse the payload as JSON
    json_data = json.loads(flask.request.data)

    # Log the request
    logging.info(json_data)

    # Parse an InferenceInput dataclass from the payload
    try:
        inputs = dacite.from_dict(InferenceInput, json_data)
    except dacite.exceptions.WrongTypeError as exception:
        logging.info(json_data)
        return str(exception), 400
    except dacite.exceptions.MissingValueError as exception:
        logging.info(json_data)
        return str(exception), 400

    response = compute_request(
        inputs=inputs,
        seed_images_dir=SEED_IMAGES_DIR,
        pipeline=PIPELINE,
    )

    # Log the total time
    logging.info(f"Request took {time.time() - start_time:.2f} s")

    return response


def compute_request(
    inputs: InferenceInput,
    pipeline: RiffusionPipeline,
    seed_images_dir: str,
) -> T.Union[str, T.Tuple[str, int]]:
    """
    Does all the heavy lifting of the request.

    Args:
        inputs: The input dataclass
        pipeline: The riffusion model pipeline
        seed_images_dir: The directory where seed images are stored
    """
    # Load the seed image by ID
    init_image_path = Path(seed_images_dir, f"{inputs.seed_image_id}.png")

    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_id}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    mask_image: T.Optional[PIL.Image.Image] = None
    if inputs.mask_image_id:
        mask_image_path = Path(seed_images_dir, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")

    # Execute the model to get the spectrogram image
    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=mask_image,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=20,
        max_frequency=20000,
    )

    # Reconstruct audio from the image
    # TODO(hayk): It may help performance a bit to cache this object
    converter = SpectrogramImageConverter(params=params, device=str(pipeline.device))

    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    # Export audio to MP3 bytes
    mp3_bytes = io.BytesIO()
    segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)

    # Export image to JPEG bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, exif=image.getexif(), format="JPEG")
    image_bytes.seek(0)

    # Assemble the output dataclass
    output = InferenceOutput(
        image="data:image/jpeg;base64," + base64_util.encode(image_bytes),
        audio="data:audio/mpeg;base64," + base64_util.encode(mp3_bytes),
        duration_s=segment.duration_seconds,
    )

    return json.dumps(dataclasses.asdict(output))


if __name__ == "__main__":
    import argh
    logging.info("riffusion started!")
    sys.stdout.flush()
    argh.dispatch_command(run_app)
