from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import librosa.feature
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from src.external.musicnn import Musicnn
from src.components.common import create_tagogram, plot_probability_graph, Config
from src.components.predictor import Predictor, SklearnPredictor

if 'selected_model_index' not in st.session_state:
    st.session_state.selected_model_index = 0

if 'data_models' not in st.session_state:
    st.session_state.data_models = {
        "MusicNN (10 classes)": Predictor(Config(model=Musicnn(n_class=10), dataset_name="mtat-10"),
                                          model_filename="mtat-10/2023-03-26-13-22-52.pth"),
        "MusicNN (20 classes)": Predictor(Config(model=Musicnn(n_class=20), dataset_name="mtat-20"),
                                          model_filename="mtat-20/2023-03-27-11-49-27.pth"),
        "KNeighborsClassifier (10 classes)": SklearnPredictor(Config(model=KNeighborsClassifier(), dataset_name="mtat-10"),
                                                              model_filename="mtat-10/model.bin"),
        "KNeighborsClassifier (20 classes)": SklearnPredictor(Config(model=KNeighborsClassifier(), dataset_name="mtat-20"),
                                                              model_filename="mtat-20/model.bin")}


def update_music_track(upload):
    st.write("## Visualisation of music tagging models")
    with NamedTemporaryFile(suffix="mp3") as temp, st.spinner('Loading...'):
        temp.write(upload.getvalue())
        temp.seek(0)

        y, sr = librosa.load(temp.name)
        fig, ax = plt.subplots(nrows=2, sharex='all')
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
        img = librosa.display.specshow(librosa.power_to_db(spectrogram), x_axis='time', y_axis='mel', ax=ax[1])
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        prediction = st.session_state.current_model.predict_tags_prob(mp3_file=temp.name)
        predicted_tags = st.session_state.current_model.predict_tags(mp3_file=temp.name)
    st.write('#### Mel-frequency spectrogram')
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    st.pyplot(fig)
    st.write(f'#### Assigned tags')
    st.write(", ".join(predicted_tags))
    st.plotly_chart(plot_probability_graph(prediction), use_container_width=True)

    # Tagogram is useful only if more than one chunk
    if prediction.shape[1] > 1:
        st.plotly_chart(create_tagogram(prediction))


if __name__ == '__main__':
    st.set_page_config(layout="centered", page_title="Music Tagging", initial_sidebar_state="expanded")
    st.sidebar.write("## :gear: Settings")
    st.sidebar.write("#### Upload/Download")
    st.sidebar.file_uploader("Please upload mp3 file.", type=['mp3'], key='uploaded_file')
    if st.session_state.uploaded_file is not None:
        st.sidebar.audio(st.session_state.uploaded_file.read(), format='audio/mp3')

    st.sidebar.write("#### Model")
    st.sidebar.selectbox(label='Select model:', options=st.session_state.data_models.keys(),
                         index=st.session_state.selected_model_index, key='selected_model_name')

    st.session_state.current_model = st.session_state.data_models[st.session_state.selected_model_name]
    print(f"Setting current model to: {st.session_state.selected_model_name}")
    if st.session_state.uploaded_file is not None:
        update_music_track(st.session_state.uploaded_file)
