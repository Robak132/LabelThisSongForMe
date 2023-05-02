import torchopenl3

from src.utils.common import convert_mp3_to_npy
from src.interfaces.base_preprocessor import BasePreProcessor


class PreProcessor(BasePreProcessor):
    def __init__(self, input_path=None, output_path=None, sr=16000, suffix=None):
        super().__init__(input_path, output_path, sr, suffix)

    def process(self, input_filename):
        return convert_mp3_to_npy(input_filename, self.sr)


class OpenL3PreProcessor(BasePreProcessor):
    def __init__(self, input_path=None, output_path=None, sr=16000, suffix=None):
        super().__init__(input_path, output_path, sr, suffix)

    def process(self, input_filename):
        x = convert_mp3_to_npy(input_filename, self.sr)
        emb, ts = torchopenl3.get_audio_embedding(x, self.sr, content_type="music", input_repr="mel256",
                                                  embedding_size=512, hop_size=1, batch_size=10, sampler="julian",
                                                  verbose=0)
        return emb.detach().cpu().numpy()
