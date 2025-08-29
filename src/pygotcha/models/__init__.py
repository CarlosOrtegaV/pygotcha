import numpy as np
from numba import njit
from typing import Tuple, Optional, List

# Step 1: Define tree structure and constants (unchanged)
# -----------------------------------------------------------------------------
TREE_LEAF: int = -1
TREE_UNDEFINED: int = -2

node_dtype = (
    ("left_child", np.int64),
    ("right_child", np.int64),
    ("feature_index", np.int64),
    ("threshold", np.float64),
    ("value", np.int64),
    ("impurity", np.float64),
    ("n_samples", np.int64),
)


# Step 2: Core Numba-jitted computational functions (unchanged)
# -----------------------------------------------------------------------------


@njit
def _gini_impurity(y: np.ndarray) -> float:
    """
    Calculate the Gini impurity for a set of labels.

    Parameters
    ----------
    y : np.ndarray
        An array of integer class labels.

    Returns
    -------
    float
        The Gini impurity of the node.
    """
    if y.size == 0:
        return 0.0

    class_0_count = np.sum(y == 0)
    class_1_count = y.size - class_0_count

    if y.size == 0:  # Redundant check but good for safety
        return 0.0

    p0 = class_0_count / y.size
    p1 = class_1_count / y.size

    gini = 1.0 - (p0**2 + p1**2)
    return gini


@njit
def _find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Find the best feature and threshold for a split.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix for the current node, shape (n_samples, n_features).
    y : np.ndarray
        The label vector for the current node, shape (n_samples,).

    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - best_feature_idx (int): The index of the best feature to split on.
        - best_threshold (float): The best threshold value for that feature.
        Returns (-1, -1.0) if no valid split is found.
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return -1, -1.0

    parent_impurity = _gini_impurity(y)
    best_gain = -1.0
    best_feature_idx = -1
    best_threshold = -1.0

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            y_left, y_right = y[left_mask], y[right_mask]

            if y_left.size == 0 or y_right.size == 0:
                continue

            w_left = y_left.size / n_samples
            w_right = y_right.size / n_samples
            children_impurity = (w_left * _gini_impurity(y_left)) + (
                w_right * _gini_impurity(y_right)
            )
            gain = parent_impurity - children_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold


@njit
def _predict_single(x_row: np.ndarray, tree: np.ndarray) -> int:
    """
    Predict the class for a single data point by traversing the tree.

    Parameters
    ----------
    x_row : np.ndarray
        A single row of features.
    tree : np.ndarray
        The trained decision tree represented as a NumPy structured array.

    Returns
    -------
    int
        The predicted class label.
    """
    node_idx = 0
    while tree[node_idx]["left_child"] != TREE_LEAF:
        feature_idx = tree[node_idx]["feature_index"]
        threshold = tree[node_idx]["threshold"]
        if x_row[feature_idx] <= threshold:
            node_idx = tree[node_idx]["left_child"]
        else:
            node_idx = tree[node_idx]["right_child"]
    return tree[node_idx]["value"]


# Step 3: The Python Class Orchestrator (MODIFIED)
# -----------------------------------------------------------------------------


class NumbaDecisionTreeClassifier:
    """
    A minimal binary decision tree classifier with max_depth.

    This implementation uses a NumPy array to store the tree structure,
    making it compatible with Numba's JIT compilation for performance.

    Parameters
    ----------
    max_depth : int, default=5
        The maximum depth of the tree.

    Attributes
    ----------
    tree_ : Optional[np.ndarray]
        The fitted tree structure, stored as a NumPy structured array.
        None if the tree has not been fitted yet.
    """

    def __init__(self, max_depth: int = 5) -> None:
        """
        Initializes the classifier.

        Raises
        ------
        ValueError
            If max_depth is not a positive integer.
        """
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")
        self.max_depth = max_depth
        self.tree_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples, shape (n_samples, n_features).
        y : np.ndarray
            The target values (class labels), shape (n_samples,).

        Raises
        ------
        ValueError
            If the input arrays are empty or have shape mismatches.
        """
        # Step 1: Input validation.
        if X.size == 0 or y.size == 0:
            raise ValueError("Input arrays X and y cannot be empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Step 2: Initialize the tree structure.
        n_samples = X.shape[0]
        max_nodes = 2 * n_samples - 1
        self.tree_ = np.zeros(max_nodes, dtype=node_dtype)

        # Step 3: Modify the stack to include depth.
        # Each tuple is (node_idx, sample_indices, current_depth).
        stack: List[Tuple[int, np.ndarray, int]] = []

        # Step 4: Start with the root node (index 0) at depth 0.
        root_indices = np.arange(n_samples)
        stack.append((0, root_indices, 0))
        node_count = 1

        # Step 5: Process the stack until it's empty.
        while stack:
            node_idx, indices, depth = stack.pop()

            X_node, y_node = X[indices], y[indices]

            # Step 6: Populate the current node's basic info.
            self.tree_[node_idx]["n_samples"] = len(indices)
            self.tree_[node_idx]["impurity"] = _gini_impurity(y_node)

            # Step 7: Check for stopping conditions.
            is_pure = self.tree_[node_idx]["impurity"] == 0.0
            is_max_depth = depth >= self.max_depth

            best_feature, best_thresh = -1, -1.0
            if not is_pure and not is_max_depth:
                best_feature, best_thresh = _find_best_split(X_node, y_node)

            # Step 8: If it's a leaf node (due to purity, max_depth, or no gain)...
            if best_feature == -1:
                self.tree_[node_idx]["left_child"] = TREE_LEAF
                self.tree_[node_idx]["right_child"] = TREE_LEAF
                self.tree_[node_idx]["value"] = np.bincount(y_node).argmax()
            else:  # Step 9: ...otherwise, it's a split node.
                self.tree_[node_idx]["feature_index"] = best_feature
                self.tree_[node_idx]["threshold"] = best_thresh
                self.tree_[node_idx]["value"] = TREE_UNDEFINED

                # Step 10: Determine children indices.
                left_mask = X_node[:, best_feature] <= best_thresh
                right_mask = ~left_mask
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]

                # Step 11: Add children to the stack with incremented depth.
                left_child_idx = node_count
                right_child_idx = node_count + 1
                self.tree_[node_idx]["left_child"] = left_child_idx
                self.tree_[node_idx]["right_child"] = right_child_idx

                stack.append((left_child_idx, left_indices, depth + 1))
                stack.append((right_child_idx, right_indices, depth + 1))
                node_count += 2

        # Step 12: Trim the tree array to the actual size used.
        self.tree_ = self.tree_[:node_count]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.

        Parameters
        ----------
        X : np.ndarray
            The input samples, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted class labels for each sample.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.tree_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            predictions[i] = _predict_single(X[i], self.tree_)
        return predictions
