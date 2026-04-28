import math
from typing import Any

import numpy as np
from pandas import DataFrame, Series


def mean(x: Series) -> float:
    return x.sum() / x.count()


def variance(x: Series) -> float:
    nominator = ((x - mean(x)) ** 2).sum()
    return nominator / (x.count() - 1)


def gauss(x: float, mean: float, std: float) -> float:
    exponent = ((x - mean) ** 2) / (2 * std ** 2)
    return np.exp(-exponent) / (std * math.sqrt(2 * np.pi))


def naive_bayes_calculate_stats(data: DataFrame, labels: Series) -> DataFrame:
    stats = data.groupby(by=labels).aggregate([mean, variance])
    stats["likelihood"] = labels.value_counts(normalize=True)
    return stats


def _naive_bayes_classify_single(stats: DataFrame, data: Series) -> Any:
    results = Series(0, index=stats.index, dtype=float)

    for label in stats.index:
        probability = math.log(stats["likelihood"][label])

        for feature in data.index:
            x = data[feature]
            mean = stats.loc[label, (feature, "mean")]
            variance = stats.loc[label, (feature, "variance")]

            probability += math.log(gauss(x, mean, math.sqrt(variance)))

        results[label] = probability

    return results.idxmax()


def naive_bayes(stats: DataFrame, unclassified: DataFrame) -> Series:
    return unclassified.apply(lambda row: _naive_bayes_classify_single(stats, row), axis=1)

# stats = naive_bayes_calculate_stats(X_train, y_train)
# y_predicted = naive_bayes(stats, X_validation)
