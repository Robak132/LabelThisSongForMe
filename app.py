from tempfile import NamedTemporaryFile

import librosa
import librosa.display
import librosa.feature
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

from src.utils.common import create_tagogram, plot_probability_graph, get_tags, create_spectrogram
from src.utils.config import Config
from src.components.predictor import Predictor, SklearnPredictor
from src.external.musicnn import Musicnn


@st.cache_resource(show_spinner="Initialising models...")
def load_models():
    musicnn_10_config = Config(model=Musicnn(n_class=10), dataset_name="mtat-10")
    musicnn_20_config = Config(model=Musicnn(n_class=20), dataset_name="mtat-20")
    knn_10_config = Config(model=KNeighborsClassifier(), dataset_name="mtat-10")
    knn_20_config = Config(model=KNeighborsClassifier(), dataset_name="mtat-20")

    return {"MusicNN (10 classes)": Predictor(musicnn_10_config, model_filename="2023-04-23-18-16-20.pth"),
        "MusicNN (20 classes)": Predictor(musicnn_20_config, model_filename="2023-03-27-11-49-27.pth"),
        "OpenL3 + K-nn Algorithm (10 classes)": SklearnPredictor(knn_10_config, model_filename="model.bin"),
        "OpenL3 + K-nn Algorithm (20 classes)": SklearnPredictor(knn_20_config, model_filename="model.bin")}


def print_music_classification_screen(upload):
    st.title("Label This Song For Me")
    with NamedTemporaryFile(suffix="mp3") as temp, st.spinner('Loading...'):
        temp.write(upload.getvalue())
        temp.seek(0)

        y, sr = librosa.load(temp.name)
        prediction = st.session_state.current_model.predict_tags_prob(mp3_file=temp.name)
    st.write('### Mel-frequency spectrogram')
    st.pyplot(create_spectrogram(y, sr))
    st.write(f'### Assigned tags')
    st.write(", ".join(get_tags(prediction)))
    st.plotly_chart(plot_probability_graph(prediction), use_container_width=True)

    # Tagogram is useful only if more than one chunk
    if prediction.shape[1] > 1:
        st.plotly_chart(create_tagogram(prediction))


def print_main_screen():
    st.title("Label This Song For Me")
    st.write("App to label songs using AI methods.")
    st.write("")
    st.image("img/main.png")
    st.write("")
    st.write("Please upload a file to classify!")


if __name__ == '__main__':
    st.set_page_config(layout="centered", page_title="Label This Song For Me", initial_sidebar_state="expanded")

    if 'selected_model_index' not in st.session_state:
        st.session_state.selected_model_index = 0

    if 'data_models' not in st.session_state:
        st.session_state.data_models = load_models()

    # Init app
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
        print_music_classification_screen(st.session_state.uploaded_file)
    else:
        print_main_screen()
