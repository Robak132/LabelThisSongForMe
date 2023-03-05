from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import librosa.feature
import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt

from models.common import Config
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
        out = tester.predict_mp3(temp.name)
        st.write(f"Best tag: {tester.tags[max((torch.argmax(out, dim=1)))]}")

        mean_out = torch.mean(out, dim=0)
        tags = [[tester.tags[i], mean_out[i].item()] for i in range(len(mean_out))]
        tags.sort(key=lambda x: x[1], reverse=True)
        tags = np.array(tags)
        print(f"Tags: {tags}")

        fig, ax = plt.subplots()
        img = ax.imshow(out.T, aspect='auto')
        ax.set_yticks(np.arange(len(tester.tags)), labels=tester.tags)
        ax.xaxis.set_visible(False)
        plt.colorbar(img, ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        tags = tags.T
        ax.barh(tags[0], tags[1], align='center')
        ax.set_yticks(np.arange(len(tags[0])), labels=tags[0])
        st.pyplot(fig)


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
