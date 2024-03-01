from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


class BinaryGradientBoostingClassifier:
    def __init__(self, base_estimator, n_trees: int, learning_rate: float):
        """
        Initializes the binary gradient boosting classifier.

        Parameters:
        - base_estimator: The base estimator to use for building the trees.
        - n_trees: The number of trees to build.
        - learning_rate: The learning rate to control the contribution of each tree.
        """
        self.base_estimator = base_estimator  # Base model to use for the ensemble
        self.n_trees = n_trees  # Number of trees in the ensemble
        self.learning_rate = learning_rate  # Learning rate for updating predictions
        self.trees: list = []  # List to store the fitted trees
        self.weights: list = []  # List to store the weights of the trees, though not used in this implementation

    @staticmethod
    def cross_entropy(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Calculate the cross-entropy loss between targets and predictions.

        Parameters:
        - target: Actual class probabilities.
        - pred: Predicted probabilities.

        Returns:
        - Cross-entropy loss values.
        """
        pred = np.clip(pred, 1e-10, 1 - 1e-10)  # Clip predictions to avoid division by zero
        return -np.log(pred) * target - (1 - target) * np.log(1 - pred)

    @staticmethod
    def cross_entropy_gradient(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the cross-entropy loss with respect to predictions.

        Parameters:
        - target: Actual class probabilities.
        - pred: Predicted probabilities, clipped to avoid division by zero.

        Returns:
        - The gradient of the loss.
        """
        pred = np.clip(pred, 1e-10, 1 - 1e-10)  # Clip predictions to avoid division by zero
        return -(target / pred) + (1 - target) / (1 - pred)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the gradient boosting model.

        Parameters:
        - X: Feature matrix.
        - y: Target values, assumed to be binary.
        """
        self.classes = np.unique(y).tolist()  # Extract unique classes from the target
        assert len(self.classes) == 2, "only binary classification is supported"
        y = np.where(y == self.classes[0], 0.0, 1.0)  # Map classes to 0 and 1
        y_pred = np.zeros_like(y)  # Initialize predictions as zeros

        # Iteratively build trees
        for _ in tqdm(range(self.n_trees)):
            y_pred_probs = np.exp(y_pred) / (1 + np.exp(y_pred))  # Convert predictions to probabilities
            gradient = self.cross_entropy_gradient(y, y_pred_probs)  # Compute the gradient of the loss
            tree = deepcopy(self.base_estimator)  # Create a deep copy of the base estimator
            tree.fit(X, gradient)  # Fit the tree to the gradient
            current_tree_preds = tree.predict(X)  # Predict with the current tree
            self.trees.append(tree)  # Store the fitted tree
            y_pred -= self.learning_rate * current_tree_preds  # Update predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
        - X: Feature matrix.

        Returns:
        - Predicted class labels.
        """
        y_preds = self.staged_predict(X)  # Get predictions from all trees
        last = y_preds[:, -1]  # Select the last set of predictions
        last = last > 0.5  # Apply threshold to determine class labels
        return np.where(last, self.classes[1], self.classes[0])  # Map predictions back to original class labels

    def staged_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels at each stage of boosting for samples in X.

        Parameters:
        - X: Feature matrix.

        Returns:
        - Predictions at each stage.
        """
        y_preds = np.zeros([len(X), self.n_trees])
        for i, tree in enumerate(tqdm(self.trees)):
            pred = tree.predict(X)
            y_preds[:, i] = y_preds[:, i - 1] - self.learning_rate * pred

        # apply sigmoid to transform log-odds -> probabilities
        y_preds = np.exp(y_preds) / (1 + np.exp(y_preds))
        return y_preds
