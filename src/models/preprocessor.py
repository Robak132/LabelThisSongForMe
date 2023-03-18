import os
import pathlib

import librosa
import numpy as np
import torchopenl3
import tqdm


def convert_mp3_to_npy(mp3_file, sr) -> np.ndarray:
    x, _ = librosa.load(mp3_file, sr=sr)
    return x


class BasePreProcessor:
    def __init__(self, input_path, output_path, suffix=None):
        self.input_path = input_path
        self.output_path = output_path
        self.suffix = suffix

    def run(self, files):
        for filename in tqdm.tqdm(files):
            input_full_filename = os.path.join(self.input_path, filename)

            if self.suffix is not None:
                filename = pathlib.Path(filename).with_suffix(f".{self.suffix}")
            output_full_filename = os.path.join(self.output_path, filename)
            if not os.path.exists(output_full_filename):
                try:
                    os.makedirs(os.path.join(*output_full_filename.split("/")[:-1]), exist_ok=True)
                    self.process(input_full_filename, output_full_filename)
                except RuntimeError and EOFError:
                    # some audio files are broken
                    print(filename)
                    continue

    def process(self, input_filename, output_filename):
        raise Exception("You cannot use abstract class")


class PreProcessor(BasePreProcessor):
    def __init__(self, input_path, output_path, sr=16000, suffix=None):
        super().__init__(input_path, output_path, suffix)
        self.sr = sr

    def process(self, input_filename, output_filename):
        x = convert_mp3_to_npy(input_filename, self.sr)
        np.save(output_filename, x)


class OpenL3PreProcessor(BasePreProcessor):
    def __init__(self, input_path, output_path, sr=16000, suffix=None):
        super().__init__(input_path, output_path, suffix)
        self.sr = sr

    def process(self, input_filename, output_filename):
        x = convert_mp3_to_npy(input_filename, self.sr)
        emb, ts = torchopenl3.get_audio_embedding(x, self.sr, content_type="music", input_repr="mel256",
                                                  embedding_size=512, hop_size=1, batch_size=10, sampler="julian",
                                                  verbose=0)
        x = emb.detach().cpu().numpy()
        np.save(output_filename, x)
