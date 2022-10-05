import os
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
from musicnn.extractor import extractor


def save_uploadedfile(uploadedfile):
    with open("raw/temp.mp3", "wb+") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


if __name__ == '__main__':
    st.title('Music tagger')
    fileObject = st.file_uploader(label="Please upload your music track.")
    if fileObject:
        save_uploadedfile(fileObject)
        taggram, tags, features = extractor("raw/temp.mp3")
        frontend_features = np.concatenate([features['temporal'], features['timbral']], axis=1)
        in_length = 3  # seconds -- by default, the model takes inputs of 3 seconds with no overlap

        # depict taggram
        plt.rcParams["figure.figsize"] = (10, 8)
        fontsize = 12
        fig, ax = plt.subplots()
        ax.imshow(taggram.T, interpolation=None, aspect="auto")

        # title
        ax.title.set_text('Taggram')
        ax.title.set_fontsize(fontsize)

        # x-axis title
        ax.set_xlabel('(seconds)', fontsize=fontsize)

        # y-axis
        y_pos = np.arange(len(tags))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tags, fontsize=fontsize - 1)

        # x-axis
        x_pos = np.arange(taggram.shape[0])
        x_label = np.arange(in_length / 2, in_length * taggram.shape[0], 3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_label, fontsize=fontsize)

        st.pyplot(fig)
