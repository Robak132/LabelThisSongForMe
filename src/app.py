from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import librosa.feature
import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt

from models.common import Config, create_tagogram, plot_probability_graph
from models.tester import Tester

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
        tester = Tester()
        raw_data, raw_tags, prediction = tester.predict_tags(mp3_file=temp.name)
        st.pyplot(plot_probability_graph(prediction))
        st.pyplot(create_tagogram(raw_data, tester.tags))


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
