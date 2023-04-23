import os
import pathlib

import numpy as np
import tqdm


class BasePreProcessor:
    def __init__(self, input_path=None, output_path=None, sr=16000, suffix=None):
        self.input_path = input_path
        self.output_path = output_path
        self.suffix = suffix
        self.sr = sr

    def run(self, files):
        for filename in tqdm.tqdm(files):
            input_full_filename = os.path.join(self.input_path, filename)

            if self.suffix is not None:
                filename = pathlib.Path(filename).with_suffix(f".{self.suffix}")
            output_full_filename = os.path.join(self.output_path, filename)
            if not os.path.exists(output_full_filename):
                try:
                    os.makedirs(os.path.join(*output_full_filename.split("/")[:-1]), exist_ok=True)
                    output = self.process(input_full_filename)
                    np.save(output_full_filename, output)
                except RuntimeError and EOFError:
                    # some audio files are broken
                    print(filename)
                    continue

    def process(self, input_filename):
        raise Exception("This method is abstract.")
