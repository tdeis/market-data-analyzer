import numpy as np


class MomentumStrategy:

    def generate_weights(self, returns, cov_matrix, context=None):

        scores = returns.mean().values

        weights = np.maximum(scores, 0)

        if weights.sum() == 0:
            return np.ones(len(weights)) / len(weights)

        return weights / weights.sum()
