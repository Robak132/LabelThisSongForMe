from utils.common import load_file_lists
from components.preprocessor import OpenL3PreProcessor

if __name__ == '__main__':
    p = OpenL3PreProcessor(input_path="../data/mtat/mp3",
                           output_path="../data/mtat/emb",
                           suffix="npy")
    p.run(files=load_file_lists(["../split/mtat/train.npy", "../split/mtat/valid.npy", "../split/mtat/test.npy"])[:, 1])
