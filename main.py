import os
import csv

import numpy as np

from utils import experiment


def run():
    def prepare_experiment_settings(num, vals):
        l = list()
        for v in vals:
            l.append(np.hstack((np.ones((num, 1)) * v, np.expand_dims(np.arange(num), 1))))
        return np.vstack(l)

    results_path = os.path.join("results", "experiments128")
    os.makedirs(results_path,  exist_ok=True)
    metrics = dict()
    args = prepare_experiment_settings(1, [50000])
    for num, idx in args:
        experiment(num, idx, 'cuda', metrics, results_path)

    w = csv.writer(open(os.path.join(results_path, "results50.csv"), "w"))
    for key, val in metrics.items():
        w.writerow([key, val])


if __name__ == '__main__':
    run()
