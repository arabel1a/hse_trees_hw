import numpy as np
import pandas as pd


def train_test_split(X: pd.DataFrame, y: pd.Series, test_frac=0.8, random_state=None):
    assert len(X) == len(y), "X and y must have the same length"
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))

    test_set_size = int(len(X) * test_frac)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def classification_report(y_true, y_pred):
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    classes = np.unique(y_true)
    report = []

    for cls in classes:
        # Redefine true positive, false positive, and false negative for the current class
        true_positive = np.sum((y_pred == cls) & (y_true == cls))
        false_positive = np.sum((y_pred == cls) & (y_true != cls))
        false_negative = np.sum((y_pred != cls) & (y_true == cls))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report.append((cls, precision, recall, f1))

    # prepare the report
    str_report = ""
    str_report += f"{'Class':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10}\n"
    str_report += "-" * 45 + "\n"
    for cls, precision, recall, f1 in report:
        str_report += f"{cls:^10} | {precision:^10.2f} | {recall:^10.2f} | {f1:^10.2f}\n"
    return str_report
