import sys

import streamlit as st
import streamlit.web.cli as stcli
from streamlit import runtime

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "2"


PAGES = {
    "🎛️ Home": "tasks.home",
    "🌊 Text to Audio": "tasks.text_to_audio",
    "✨ Audio to Audio": "tasks.audio_to_audio",
    "✨ Audio to Audio Batch": "tasks.audio_to_audio_batch",
    "🎭 Interpolation": "tasks.interpolation",
    "✂️ Audio Splitter": "tasks.split_audio",
    "📜 Text to Audio Batch": "tasks.text_to_audio_batch",
    "📎 Sample Clips": "tasks.sample_clips",
    "⏈ Spectrogram to Audio": "tasks.image_to_audio",
}


def render() -> None:
    st.set_page_config(
        page_title="Riffusion Playground",
        page_icon="🎸",
        layout="wide",
    )

    page = st.sidebar.selectbox("Page", list(PAGES.keys()))
    assert page is not None
    module = __import__(PAGES[page], fromlist=["render"])
    module.render()


if __name__ == "__main__":
    if runtime.exists():
        render()
    else:
        sys.argv = ["streamlit", "run"] + sys.argv
        sys.exit(stcli.main())
