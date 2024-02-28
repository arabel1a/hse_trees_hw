from copy import deepcopy

import numpy as np
import pandas as pd


class RandomForest:
    def __init__(self, base_estimator, n_trees: int, seed=123):
        np.random.seed(seed)
        self.n_trees = n_trees
        self.base_estimator = base_estimator

    def fit(self, features, target):
        self.n_features = features.shape[1]
        self.x = features
        self.y = target
        self.trees = [self.create_tree() for i in range(self.n_trees)]

    def create_tree(self):
        # Divide databse into small dataset
        sample_count = self.y.shape[0]
        idx = np.random.choice(
            np.arange(sample_count),
            size=sample_count,
            replace=True,
        )

        y = self.y.iloc[idx].copy()
        x = self.x.iloc[idx].copy()

        model = deepcopy(self.base_estimator)
        model.fit(x, y)

        return model

    def predict(self, x):
        prediction = []

        for dt in self.trees:
            prediction.append(dt.predict(x)[None, ...])

        preds = np.concatenate(prediction, 0)
        preds = pd.DataFrame(preds.T).mode(axis=1)
        return preds
