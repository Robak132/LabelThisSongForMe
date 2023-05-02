import os
import pathlib
import random

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from utils.common import load_file_lists
from components.preprocessor import OpenL3PreProcessor


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    split_path = "../split/mtat-20/"
    os.makedirs(split_path, exist_ok=True)
    set_seed(123456)

    p = OpenL3PreProcessor(input_path="../data/mtat/mp3",
                           output_path="../data/mtat/emb",
                           suffix="npy")

    data = load_file_lists([
        os.path.join(split_path, "train.npy"),
        os.path.join(split_path, "valid.npy"),
        os.path.join(split_path, "test.npy")
    ])
    p.run(files=data[:, 1])
    binary = {row[0]: row[1:] for row in np.load(os.path.join(split_path, "binary.npy"), allow_pickle=True)}
    tags = np.load(os.path.join(split_path, "tags.npy"), allow_pickle=True)

    X = []
    Y = []
    for idx, filename in data:
        filename = os.path.join("../data/mtat/emb", str(pathlib.Path(filename).with_suffix(".npy")))
        file_data = np.load(filename, allow_pickle=True).flatten()
        X.append(file_data)
        Y.append(binary[int(idx)])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    # model = RandomForestClassifier(bootstrap=True,
    #                                max_depth=20,
    #                                max_features='sqrt',
    #                                n_jobs=4,
    #                                random_state=1,
    #                                warm_start=True)
    # model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_pred, Y_test))
    #
    # model = DecisionTreeClassifier(max_depth=20, max_features='sqrt', random_state=1)
    # model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_pred, Y_test))
    #
    # model = MultiOutputClassifier(LogisticRegression(solver='lbfgs', max_iter=1000), n_jobs=4)
    # model.partial_fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_pred, Y_test))

    model = MultiOutputClassifier(SVC(max_iter=1000), n_jobs=4)
    model.partial_fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test))