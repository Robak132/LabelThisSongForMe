import numpy as np
import pandas as pd

if __name__ == '__main__':
    adnotations = pd.read_csv('../data/mtat/annotations_final.csv', sep='\t')
    tags_table = adnotations.loc[:, adnotations.columns != 'clip_id'].T.sum()
    clip_info = pd.read_csv('../data/mtat/clip_info_final.csv', sep='\t')

    train = np.load("../split/mtat/train.npy")
    valid = np.load("../split/mtat/valid.npy")
    binary = np.load("../split/mtat/binary.npy")
    tags = np.load("../split/mtat/tags.npy")
    test = np.load("../split/mtat/test.npy")
    print("a")
