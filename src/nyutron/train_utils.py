import numpy as np


class Subsampler:
    """subsample huggingface dataset with a given seed"""

    def __init__(self, seed, data) -> None:
        self.seed = seed
        self.data = data
        self.total = len(data)
        self.total_indices = np.arange(self.total)
        self.rng = np.random.default_rng(seed)

    def subsample(self, n_samples):
        print(f"subsampling {n_samples} data")
        # reference: https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
        indices = self.rng.choice(self.total_indices, size=n_samples, replace=False)
        samples = self.data.select(indices)
        return samples
