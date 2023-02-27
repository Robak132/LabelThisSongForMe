import os
from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import librosa.feature
import streamlit as st
import torch
from matplotlib import pyplot as plt

from models.provider import Provider

DATA_MODELS = {
    "MusiCNN": "../models/musicnn.pth",
    "EdgeL3": "../models/musicnn.pth"
}


def update_music_track(upload):
    st.sidebar.audio(upload.read(), format='audio/mp3')
    with NamedTemporaryFile(suffix="mp3") as temp:
        temp.write(upload.getvalue())
        temp.seek(0)

        st.write('#### Mel-frequency spectrogram')
        y, sr = librosa.load(temp.name)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.waveshow(y, sr=sr, ax=ax[0])

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
        img = librosa.display.specshow(librosa.power_to_db(spectrogram), x_axis='time', y_axis='mel', ax=ax[1])
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        st.pyplot(fig)

        st.write('#### Tags')
        provider = Provider("models/musicnn.pth")
        data = provider.get_tags(temp.name)
        st.write(f"Best tag: {provider.tags[max((torch.argmax(data, dim=1)))]}")

def setup_sidebar():
    st.sidebar.write("## :gear: Settings")
    st.sidebar.write("#### Upload/Download")
    uploaded_file = st.sidebar.file_uploader("Please upload mp3 file.", type=['mp3'])
    if uploaded_file is not None:
        update_music_track(uploaded_file)
    st.sidebar.write("#### Model")
    return DATA_MODELS[st.sidebar.selectbox('Select model', DATA_MODELS.keys())]


if __name__ == '__main__':
    st.set_page_config(layout="centered", page_title="Music Tagging", initial_sidebar_state="expanded")
    st.write("## Visualisation of music tagging models")

    model = setup_sidebar()
