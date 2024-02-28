import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import xlogy


class Node:
    def __init__(self):
        self.type = None
        self.height = 1
        self.entropy = None
        self.n_samples = None

        # if splitting
        self.IG = None
        self.split_feature_id = None
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None

        # if terminating
        self.label = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        # initialize tree paramters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        # check inputs
        if isinstance(X, pd.DataFrame):
            self.X = X.values
            self.features = X.columns
        elif isinstance(X, np.ndarray):
            if len(X.shape) != 2:
                raise ValueError("X should be of shape (n_objects, n_features)")
            self.X = X
            self.features = np.arange(0, X.shape[1])
        else:
            raise TypeError("X should be pandas DataFrame or numpy ndarray")

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            self.y = y.values
        elif isinstance(y, np.ndarray):
            if len(X.shape) != 2:
                raise ValueError("y should be of shape (n_objects,)")
            self.y = y
        else:
            raise TypeError("y should be pandas DataFrame or numpy ndarray")

        idx = np.arange(self.X.shape[0])
        feature_ids = np.arange(self.X.shape[1])
        self.node = None

        # start recursion
        self.node = self._fit_rec(idx, feature_ids, self.node)

    def _entropy(self, idx):
        # count occurencies of each class
        count = np.unique(self.y[idx], return_counts=True)[1]
        # calculate probabilities of each class
        p = count / idx.size
        # calculate entropy
        entropy = -(xlogy(p, p) / np.log(2)).sum()
        return entropy

    def _cond_entropy(self, idx, feature_id):
        # note that here we buil binary tree
        # if some feature takes multiple values
        # we try splitting the tree as follows
        #
        #           target_feature == value
        #                  /   \
        #             yes /     \ no
        #                /       \
        #               V         V
        #
        # and not by bultiple features
        #               target_feature ==
        #              /    /    \    \
        #             /     |     |    \
        #             V     V     V     V
        #          val_1  val_2  val_3  val_4

        # get feature values and corresponding counts of target feature
        feature_vals, val_counts = np.unique(
            self.X[idx, feature_id],
            return_counts=True,
        )
        P = val_counts / idx.size
        # calculate probabilities of each feature value of target feature

        # indices, where target feature is equal to each of the feature values
        equal_idx = [np.intersect1d(np.where(self.X[:, feature_id] == val)[0], idx) for val in feature_vals]
        # indices, where target feature is not equal to each of the feature values
        n_equal_idx = [np.intersect1d(np.where(self.X[:, feature_id] != val)[0], idx) for val in feature_vals]

        # conditional entropies of each value of each feature value
        H_cond = [
            self._entropy(eq_idx) * p + self._entropy(neq_idx) * (1 - p)
            for (eq_idx, neq_idx, p) in zip(equal_idx, n_equal_idx, P)
        ]

        # get best feature value
        best_id = np.argmax(H_cond)
        return (
            feature_vals[best_id],
            H_cond[best_id],
            equal_idx[best_id],
            n_equal_idx[best_id],
        )

    def _info_gain(self, idx, feature_id):
        # claculate information gain if split target feature
        H = self._entropy(idx)

        (
            best_split_val,
            best_H_cond,
            best_equal_idx,
            best_n_equal_idx,
        ) = self._cond_entropy(idx, feature_id)

        best_IG = H - best_H_cond
        return best_split_val, best_IG, best_equal_idx, best_n_equal_idx

    def _get_best_feature(self, idx, feature_ids):
        # get information gains of all features
        best_split_vals, best_IGs, best_equal_idx, best_n_equal_idx = map(
            list,
            zip(
                *[self._info_gain(idx, feature_id) for feature_id in feature_ids],
            ),
        )

        best_feature_idx = np.argmax(best_IGs)

        return (
            feature_ids[best_feature_idx],
            best_split_vals[best_feature_idx],
            best_IGs[best_feature_idx],
            best_equal_idx[best_feature_idx],
            best_n_equal_idx[best_feature_idx],
        )

    def _fit_rec(self, idx, feature_ids, node):
        if not node:
            node = Node()
            node.height = 1

        node.n_samples = idx.size
        # terminating cases
        # 1) all labels left are the same
        if np.unique(self.y[idx]).size == 1:
            node.type = "terminal"
            node.label = self.y[idx][0]
            desc = "N: {}\nclass: {}".format(
                node.n_samples,
                node.label,
            )
            node.desc = desc
            return node

        # 2) no more features to split or not enough samples to split
        if feature_ids.size == 0 or node.n_samples < self.min_samples_split or node.height == self.max_depth:

            node.type = "terminal"
            left_labels, label_counts = np.unique(self.y[idx], return_counts=True)
            best_label = left_labels[np.argmax(label_counts)]
            node.label = best_label
            desc = "N: {}\nclass: {}".format(
                node.n_samples,
                node.label,
            )
            node.desc = desc
            return node

        (
            best_feature_id,
            best_split_value,
            best_IG,
            best_equal_idx,
            best_n_equal_idx,
        ) = self._get_best_feature(idx, feature_ids)

        node.type = "internal"
        node.entropy = self._entropy(idx)
        node.IG = best_IG
        node.split_value = best_split_value
        node.split_feature_id = best_feature_id
        node.split_feature = self.features[best_feature_id]
        desc = "{} == {}\nentropy: {:.2f}\nN: {}".format(
            node.split_feature,
            node.split_value,
            node.entropy,
            node.n_samples,
        )
        node.desc = desc

        left = Node()
        left.height = node.height + 1
        # eliminate feature if all values of target feature are the same in child node
        feature_ids_left = feature_ids
        if len(np.unique(self.X[best_equal_idx, best_feature_id])) == 1:
            feature_ids_left = np.delete(
                feature_ids,
                np.where(feature_ids == best_feature_id)[0],
            )
        left = self._fit_rec(best_equal_idx, feature_ids_left, left)
        node.left = left

        right = Node()
        right.height = node.height + 1
        # eliminate feature if all values of target feature are the same in child node
        feature_ids_right = feature_ids
        if len(np.unique(self.X[best_n_equal_idx, best_feature_id])) == 1:
            feature_ids_right = np.delete(
                feature_ids,
                np.where(feature_ids == best_feature_id)[0],
            )
        right = self._fit_rec(best_n_equal_idx, feature_ids_right, right)
        node.right = right
        return node

    def _predict(self, item, node):
        if node.type == "terminal":
            return node.label
        if item[node.split_feature_id] == node.split_value:
            return self._predict(item, node.left)
        else:
            return self._predict(item, node.right)

    def predict(self, X):
        # check input
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, np.ndarray):
            if len(X.shape) != 2:
                raise ValueError("X should be of shape (n_objects, n_features)")
        else:
            raise TypeError("X should be pandas DataFrame or numpy ndarray")

        ans = np.zeros(X.shape[0], dtype=self.y.dtype)
        for i in range(X.shape[0]):
            ans[i] = self._predict(X[i], self.node)

        return ans

    def plot_tree(self):
        self.G = nx.DiGraph()
        _ = self._traverse(self.node, 0)
        pos = self._hierarchy_pos(self.G, 0)
        labels = nx.get_node_attributes(self.G, "desc")
        plt.figure(figsize=(20, 20))
        nx.draw_networkx(
            self.G,
            pos=pos,
            labels=labels,
            arrows=True,
            bbox=dict(facecolor="white"),
        )

    def _hierarchy_pos(self, G, root, levels=None, width=1.0, height=1.0):
        TOTAL = "total"
        CURRENT = "current"

        def make_levels(levels, node=root, currentLevel=0, parent=None):
            """Compute the number of nodes for each level"""
            if currentLevel not in levels:
                levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
            levels[currentLevel][TOTAL] += 1
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if not neighbor == parent:
                    levels = make_levels(levels, neighbor, currentLevel + 1, node)
            return levels

        def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
            dx = 1 / levels[currentLevel][TOTAL]
            left = dx / 2
            pos[node] = ((left + dx * levels[currentLevel][CURRENT]) * width, vert_loc)
            levels[currentLevel][CURRENT] += 1
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if not neighbor == parent:
                    pos = make_pos(
                        pos,
                        neighbor,
                        currentLevel + 1,
                        node,
                        vert_loc - vert_gap,
                    )
            return pos

        if levels is None:
            levels = make_levels({})
        else:
            levels = {level: {TOTAL: levels[level], CURRENT: 0} for level in levels}
        vert_gap = height / (max([level for level in levels]) + 1)
        return make_pos({})

    def _traverse(self, node, node_num):

        if node.type == "terminal":
            self.G.add_node(
                node_num,
                type=node.type,
                n_samples=node.n_samples,
                label=node.label,
                desc=node.desc,
            )
            return node_num
        else:
            self.G.add_node(
                node_num,
                type=node.type,
                entropy=node.entropy,
                n_samples=node.n_samples,
                info_gain=node.IG,
                split_feature=node.split_feature,
                split_value=node.split_value,
                desc=node.desc,
            )

            self.G.add_node(node_num + 1)
            self.G.add_edge(node_num, node_num + 1)
            last_num = self._traverse(node.left, node_num + 1)

            self.G.add_node(last_num + 1)
            self.G.add_edge(node_num, last_num + 1)
            last_num = self._traverse(node.right, last_num + 1)

            return last_num
