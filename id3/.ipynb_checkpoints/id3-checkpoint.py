import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from networkx.drawing.nx_pydot import graphviz_layout
from queue import Queue
from typing import Any


class TreeNode:
    def __init__(self, attribute: str, ig:float, parent=None, parent_attribute_value=None, lvl=0):
        """
        Args:
        - attribute (str): the name of the feature, by which the dataset will be divided
        - ig (float): information gain
        - parent (TreeNode, optional) - the parent node of the current node
        - parent_attribute__value (Any): the value of the feature, by which the dataset was divided
        - lvl (int): the depth of the given tree (starts from 1)
        """
        self.attribute = attribute
        self.ig = ig
        self.parent = parent
        self.parent_attribute_value = parent_attribute_value
        self.lvl=lvl
        self.children = []
        self.children_attributes = []
    
    def set_prediction(self, prediction):
        """
        Sets node predictions
        """
        self.prediction = prediction
    
    def get_prediction(self):
        """
        Returns node predictions
        """
        return self.prediction

    def print_tree(self):
        """
        Prints the structure of the tree. 
        """
        space = '  ' * self.lvl
        if not self.children:
            print(f'{space}lvl#{self.lvl}: '
                  f'parent_attr_val = {self.parent_attribute_value}, '
                  f'prediction = {self.prediction.values}')
        else:
            print(f'{space}lvl#{self.lvl}: attribute = {self.attribute}, '
                  f'parent_attr_val = {self.parent_attribute_value}')
        for child in self.children:
            child.print_tree()

class DecisionTree:
    def __init__(self):
        pass       

    @staticmethod
    def calculate_target_entropy(df: pd.DataFrame, target) -> float:
        """
        Calculate the entropy of the target variable.

        Args:
        - df (pd.DataFrame): Target variable.

        Returns:
        - float: Target entropy value.
        """

        grouped = df.groupby(target).agg({'__weight':'sum'})
        
        entropy = 0
        # value_counts = vector.value_counts()
        total = df['__weight'].sum()    
        
        for _, row in grouped.iterrows():
            p = row['__weight'] / total
            entropy += -p * np.log(p)
        return entropy
       
    # Calculate E(Target | Attribute) - Entropy of Features
    def calculate_attribute_entropy(self, dataframe: pd.DataFrame, attribute: str, target: str) -> float: 
        """
        Calculate the entropy of features based on a specific attribute.

        Args:
        - dataset (pd.DataFrame): Input dataset.
        - attribute (str): Attribute to calculate entropy for.
        - target (str): Target variable.

        Returns:
        - float: Attribute entropy value.
        """
        obj_num = dataframe.shape[0]
        attribute_values = dataframe[attribute].unique()
        entropy = 0
        for value in attribute_values:
            dataframe_ = dataframe[dataframe[attribute] == value]
            value_entropy = self.calculate_target_entropy(dataframe_, target)
            p = dataframe_.shape[0] / obj_num 
            entropy += p * value_entropy
        return entropy

    def calculate_information_gain(self, dataframe: pd.DataFrame, attribute: str, target: str) -> float:
        """
        Calculate information gain for a specific attribute.

        Args:
        - dataset (pd.DataFrame): Input dataset.
        - attribute (str): Attribute to calculate information gain for.
        - target (str): Target variable.

        Returns:
        - float: Information gain value.
        """
        if attribute == target:
            return 0
        target_entropy = self.calculate_target_entropy(dataframe, target)
        attribute_entropy = self.calculate_attribute_entropy(dataframe, attribute, target)
        return target_entropy - attribute_entropy

    def winner_attribute(self, df: pd.DataFrame, target: str) -> str:
        """
        Determine the decision node by calculating the maximum information gain.

        Args:
        - df (pd.DataFrame): Input dataset.
        - target str: name of target column

        Returns:
        - str: Winning feature (decision node).
        - str: maximum information gain
        """
        max_ig = -1
        winner_feature = ""
        for column in df.columns:
            if column == target:
                continue
            ig = self.calculate_information_gain(
                df, column, target
            )
            if ig > max_ig:
                max_ig = ig
                winner_feature = column
        return winner_feature, max_ig
    
        
    def split_dataset(self, df: pd.DataFrame, attribute: str, value: Any) -> pd.DataFrame:
        """
        Split the dataset based on the decision node.

        Args:
        - df (pd.DataFrame): Input dataset.
        - attribute (str): Attribute, by which the dataset is divided.
        - value (Any): Value of the attribute.

        Returns:
        - pd.DataFrame: Subdataset after splitting.
        """
        return df[df[attribute] == value].reset_index(drop=True)
            
    def build_tree(self, df: pd.DataFrame, target: str, weight:str=None, max_depth=100) -> Any:
        """
        Args:
        - df (pd.DataFrame): Input dataset.
        - target (str): the name of  the target attribute

        Returns:
        - root (TreeNode): root of the DecisionTree
        """
        if weight is None:
            df['__weight'] = 1.
        else:
            df['__weight'] = df[weigth]
        attribute, ig = self.winner_attribute(df, target)
        root = TreeNode(attribute, ig)
        q = Queue()
        q.put(root)
        parent_partitions = Queue()
        parent_partitions.put(df)
        while not q.empty():
            current = q.get()
            partition = parent_partitions.get()
            unique_attr = partition[current.attribute].unique()
            if current.ig > 0:
                for u in unique_attr:
                    tmp = self.split_dataset(partition, attribute=current.attribute, value=u)
                    tmp.drop(current.attribute, axis=1, inplace=True)
                    attribute, ig = self.winner_attribute(tmp, target)
                    child = TreeNode(attribute, ig, current, u, current.lvl+1)
                    current.children.append(child)
                    current.children_attributes.append(u)
                    if current.lvl < max_depth and  tmp.shape[0] > 1 and current.lvl < (df.shape[1]-2):
                        q.put(child)
                        parent_partitions.put(tmp)
                    else:
                        child.set_prediction(tmp[target])
            else:
                current.set_prediction(partition[target])
        return root
    
    # Start training process. Ultimate goal is to make a decision tree.
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Start the training process to build the decision tree.

        Args:
        - df (pd.DataFrame): Input dataset.
        """
        self.tree = self.build_tree(df, **kwargs) 
        return

    def traverse_tree(self, x_test: pd.Series, root: TreeNode) -> Any:
        """
        Traverse through the decision tree.

        Args:
        - x_test (pd.Series): Input values.
        - root (TreeNode): Decision tree root.

        Returns:
        - Any: Prediction result.
        """
        node = root
        value = x_test[node.attribute]
        while node.children:
            value = x_test[node.attribute]
            for ch, att in zip(node.children, node.children_attributes):
                if(att == value):
                    node = ch
                    break
        return node.get_prediction() 
    
    def predict(self, X_test: pd.DataFrame, undefined_value='Undefined') -> Any:
        """
        Predict the class using input values.

        Args:
        - X_test (pd.DataFrame): Input values.
        - undefined_value (Any, optional): The value to put when prediction is ambiguous.

        Returns:
        - pd.DataFrame: DataFrame of predicted classes.
        
        """
        if hasattr(self, 'tree'):
            tree = self.tree
        else:
            print('This DecisionTree instance is not fitted yet.')
            return
        prediction = X_test.apply(self.traverse_tree, root=tree, axis=1)
        if prediction.shape[1] > 1:
            undefined = ~prediction.isna().any(axis=1)
            prediction[undefined] =  undefined_value
        return prediction[0]
    
    def check_tree(self):
        """
        Prints the structure of the tree.
        """
        if hasattr(self, 'tree'):
            self.tree.print_tree()
        else:
            print('This DecisionTree instance is not fitted yet.')



def draw_tree(tree: DecisionTree, fig_w=10, fig_h=5, arrow_size=20, font_size=12):
    """
    Plots given DecisionTree.
    Args:
     - tree (DecisionTree): the tree to draw
     - fig_w (int, optional): figure width
     - fig_h (int, optional): figure height
     - arrow_size (int, optional): size of the arrow end
     - font_size (int, optional): fort size of the labels
    """
    G = nx.DiGraph()
    q = Queue()
    q.put(tree)
    edge_labels = {}
    node_labels = {}
    while not q.empty():
        current = q.get()
        if(current.children):
            node_labels[current] = f'IG = {current.ig:0.3f}\n{current.attribute}'
        else:
            node_labels[current] = dict(current.get_prediction().value_counts())
        for ch in current.children:
            edge_labels[current, ch] = ch.parent_attribute_value
            G.add_edge(current, ch)
            q.put(ch)
    plt.figure(figsize=(fig_w, fig_h));
    pos = nx.nx_agraph.graphviz_layout(G, 'dot') 
    nx.draw(
        G, pos, node_shape='', edge_color='k', 
        arrowsize=arrow_size, arrowstyle='->'
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, 
        font_color='r', font_size=font_size
    )
    box = dict(facecolor='w', edgecolor='black', boxstyle='square, pad=0.2')
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, 
        bbox=box, font_size=font_size
    );