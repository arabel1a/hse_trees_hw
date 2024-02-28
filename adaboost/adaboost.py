from copy import deepcopy

import numpy as np
import pandas as pd


class BinaryAdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=3, learning_rate=1.0, **kwargs):
        self.n_estimators = n_estimators
        self.estimators = None
        self.estimator_weights = None
        self.kwargs = kwargs
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.classes = np.unique(y).tolist()
        assert len(self.classes) == 2
        #  resetting lists
        self.estimators = []
        self.estimator_weights = []

        # initialise equal weights
        sample_weights = np.ones_like(y)
        sample_weights /= len(y)

        for est_i in range(self.n_estimators):
            # 1. fit weak estimator
            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, y)

            # 2. find e
            predictions = estimator.predict(X)
            e = np.sum(
                np.where(
                    predictions != y,
                    sample_weights,
                    0,
                ),
            )

            # 3.find alpha
            alpha = np.log((1 - e) / (e + 1e-10)) / 2

            # 4. recompute w_i
            sample_weights = sample_weights * np.exp(
                ((predictions == y).astype(int) * 2 - 1) * alpha,
            )

            # 4.5 save estimator
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)

            # 5. renormalize weights
            sample_weights = sample_weights / sample_weights.sum()

    def predict(self, X, verbose=False):
        # Initialize array to hold the weighted sum of predictions
        weighted_sum = np.zeros(len(X))

        # Iterate over each estimator and its weight
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            # Predict with the current estimator
            predictions = estimator.predict(X)
            predictions = np.where(predictions == self.classes[0], 1, -1)

            # Update the weighted sum with the current predictions and weight
            weighted_sum += weight * predictions

        # Compute the final predictions by sign of the weighted sum
        final_predictions = np.where(weighted_sum > 0, self.classes[0], self.classes[1])

        return final_predictions
