import csv
import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from utils.config import Config
from components.tester import SklearnTester

if __name__ == '__main__':
    with open('openl3.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["name", "size", "f1_score", "roc_auc", "pr_auc"])

    for n in [10, 20]:
        sklearn_config = Config(model=KNeighborsClassifier(),
                                model_filename_path = "../models",
                                data_path='../data',
                                dataset_split_path= "../split",
                                dataset_name=f"mtat-{n}")

        for model_filename in glob.glob(f"../models/KNeighborsClassifier/mtat-{n}/*"):
            model_filename = Path(model_filename).name
            print(model_filename)

            tester = SklearnTester(sklearn_config, model_filename=model_filename)
            stats = tester.test()
            est_bin_array = np.where(stats.est_array >= 0.5, 1, 0)
            print(classification_report(stats.gt_array, est_bin_array, target_names=list(tester.predictor.tags), zero_division=1))

            with open('openl3.csv', 'a') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([model_filename, n, stats.f1_score, stats.roc_auc, stats.pr_auc])

