from typing import Any

import numpy as np
from pandas import DataFrame, Series


def series_entropy(data: Series) -> float:
    probabilities = data.value_counts(normalize=True)
    return -(probabilities * np.log2(probabilities)).sum()


def _c45_calculate_threshold(values: Series, labels: Series) -> tuple[float, float]:
    sorted_df = DataFrame({"value": values, "label": labels}).sort_values(by="value")

    values = sorted_df["value"].values
    labels = sorted_df["label"]

    n = len(sorted_df)

    labels_entropy = series_entropy(labels)

    best_split = (-1, 0)

    for i in range(1, n):
        if values[i - 1] == values[i]:
            continue

        left_entropy = series_entropy(labels[:i])
        right_entropy = series_entropy(labels[i:])

        left_probability = i / n
        right_probability = 1 - left_probability

        weighted_sides_entropy = left_probability * left_entropy + right_probability * right_entropy

        entropy_gain = labels_entropy - weighted_sides_entropy
        split_entropy = -(left_probability * np.log2(left_probability) + right_probability * np.log2(right_probability))

        entropy_gain_ratio = entropy_gain / split_entropy

        if entropy_gain_ratio > best_split[0]:
            threshold = (values[i - 1] + values[i]) / 2
            best_split = (entropy_gain_ratio, threshold)

    return best_split


def c45_build_tree(data: DataFrame, labels: Series, max_depth: int | None = None) -> dict:
    if len(data) < 1:
        return {"leaf": True}

    if len(data) < 2:
        return {"leaf": True, "value": labels.iloc[0]}

    if max_depth is not None:
        if max_depth < 1:
            return {"leaf": True, "value": labels.mode().iloc[0]}
        else:
            max_depth -= 1

    best_feature = ("", -1, 0)

    for column in data.columns:
        gain, threshold = _c45_calculate_threshold(data[column], labels)

        if gain <= best_feature[1]:
            continue

        best_feature = column, gain, threshold

    column, gain, threshold = best_feature

    if gain <= 0:
        return {"leaf": True, "value": labels.mode().iloc[0]}

    left = data[data[column] < threshold]
    right = data[data[column] > threshold]

    return {"leaf": False, "feature": column, "threshold": threshold,
            "left": c45_build_tree(left, labels[left.index], max_depth),
            "right": c45_build_tree(right, labels[right.index], max_depth)}


def _c45_classify_single(tree: dict, row: Series) -> Any:
    node = tree

    while node and not node["leaf"]:
        value = row[node["feature"]]

        if value < node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]

    if node and node["leaf"]:
        return node["value"]

    return None


def c45_classify(tree: dict, data: DataFrame) -> Series:
    return data.apply(lambda row: _c45_classify_single(tree, row), axis=1)

# tree = c45_build_tree(X_train, y_train)
# y_predicted = c45_classify(tree, X_validation)
