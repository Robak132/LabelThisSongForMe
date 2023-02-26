from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

DATA_MODELS = {
    "MusiCNN": "models/musicnn.pth",
    "EdgeL3": "models/musicnn.pth"
}


def update_music_track(upload):
    st.sidebar.audio(upload.read(), format='audio/mp3')
    st.write(upload.name)

    with NamedTemporaryFile(suffix="mp3") as temp:
        temp.write(upload.getvalue())
        temp.seek(0)

        st.write('#### Mel-frequency spectrogram')
        y, sr = librosa.load(temp.name)
        plot = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
        plot = librosa.power_to_db(plot, ref=np.max)
        img = librosa.display.specshow(plot, y_axis='mel', x_axis='time')
        plt.colorbar()
        plt.tight_layout()
        st.pyplot()


def setup_sidebar():
    st.sidebar.write("## :gear: Settings")
    st.sidebar.write("#### Upload/Download")
    uploaded_file = st.sidebar.file_uploader("Please upload mp3 file.", type=['mp3'])
    if uploaded_file is not None:
        update_music_track(uploaded_file)
    st.sidebar.write("#### Model")
    return DATA_MODELS[st.sidebar.selectbox('Select data model', DATA_MODELS.keys())]


if __name__ == '__main__':
    st.set_page_config(layout="centered", page_title="Music Tagging")
    st.write("## Visualisation of music tagging models")

    model = setup_sidebar()
