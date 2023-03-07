import torchopenl3

from models.common import convert_mp3_to_npy

audio_filepath = '../../data/mtat/mp3/2/aba_structure-tektonik_illusion-01-terra-59-88.mp3'
audio = convert_mp3_to_npy(audio_filepath, sr=16000)
emb, ts = torchopenl3.get_audio_embedding(audio, 16000, content_type="music", input_repr="linear", embedding_size=512, batch_size=10)
print(emb)