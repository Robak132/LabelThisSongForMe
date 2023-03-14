import os
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.common import load_file_lists

if __name__ == '__main__':
    data = load_file_lists(["../split/mtat/train.npy", "../split/mtat/valid.npy", "../split/mtat/test.npy"])
    binary = np.load("../split/mtat/binary.npy", allow_pickle=True)

    X = []
    Y = []
    for idx, filename in data:
        filename = os.path.join("../data/mtat/emb", str(pathlib.Path(filename).with_suffix(".npy")))
        X.append(np.load(filename, allow_pickle=True))
        Y.append(binary[idx])

    rfc = RandomForestClassifier(bootstrap=True,
                                 max_depth=10,
                                 max_features='sqrt',
                                 random_state=1)
    rfc.fit(X, Y)
