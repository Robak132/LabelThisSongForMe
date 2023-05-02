import csv
import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from utils.config import Config
from components.tester import Tester
from external.musicnn import Musicnn

if __name__ == '__main__':
    with open('musicnn.csv', 'w+') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["name", "size", "f1_score", "roc_auc", "pr_auc"])

    for n in [10, 20]:
        musicnn_config = Config(model=Musicnn(n_class=n), model_filename_path="../models", data_path='../data',
                                dataset_split_path="../split", dataset_name=f"mtat-{n}")

        for model_filename in glob.glob(f"../models/Musicnn/mtat-{n}/*"):
            model_filename = Path(model_filename).name
            print(model_filename)

            tester = Tester(musicnn_config, model_filename=model_filename)
            stats = tester.test()
            est_bin_array = np.where(stats.est_array >= 0.5, 1, 0)
            print(classification_report(stats.gt_array, est_bin_array, target_names=list(tester.predictor.tags), zero_division=1))

            with open('musicnn.csv', 'a') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([model_filename, n, stats.f1_score, stats.roc_auc, stats.pr_auc])

