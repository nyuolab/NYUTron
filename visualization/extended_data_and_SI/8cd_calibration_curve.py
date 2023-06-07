from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

normalizes = [True, False]
splits = ["prospective", "temporal"]
for split, normalize in product(splits, normalizes):
    y_true = np.load(f"nyutron_{split}_labels.npy")
    y_pred = np.load(f"nyutron_{split}_probs.npy")[:, 1]
    if normalize:
        y_pred = y_pred / y_pred.max()
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker=".", label="NYUTron")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideally Calibrated")
    plt.legend()
    plt.xlabel("Average Predicted Probability in each bin")
    plt.ylabel("Ratio of positives")
    plt.savefig(f"slurm_outs/calibration_curve_{split}_norm{normalize}.png", dpi=300)
