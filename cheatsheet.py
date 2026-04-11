import random
from typing import Callable

import numpy as np
from pandas import DataFrame, Series


def shuffle_dataset(data: DataFrame) -> None:
    for i in range(len(data)):
        random_id = random.randint(0, len(data) - 1)
        data.iloc[i], data.iloc[random_id] = data.iloc[random_id], data.iloc[i]


def split_dataset(data: DataFrame, percentage: float) -> tuple[DataFrame, DataFrame]:
    assert 0 <= percentage <= 1

    take = int(len(data) * percentage)
    return data[:take], data[take:]


def euclidean_distance(data: DataFrame, centroid: Series) -> Series:
    return ((data - centroid) ** 2).sum(axis=1) ** 0.5


def minkowski_distance(data: DataFrame, centroid: Series, p=4) -> Series:
    return ((data - centroid).abs() ** p).sum(axis=1) ** (1 / p)


def manhattan_distance(data: DataFrame, centroid: Series) -> Series:
    return minkowski_distance(data, centroid, p=1)


def k_means(data: DataFrame, k: int, metric: Callable[[DataFrame, Series], Series], max_iterations: int = 10) -> Series:
    assert max_iterations > 0

    centroids = data.sample(k).reset_index(drop=True)

    for _ in range(max_iterations):
        distances = DataFrame(0, index=data.index, columns=centroids.index)

        for centroid_id, centroid in centroids.iterrows():
            distances[centroid_id] = metric(data, centroid)

        clusters = distances.idxmin(axis=1)
        new_centroids = data.groupby(clusters).mean().reset_index(drop=True)

        if len(new_centroids) < k:
            break

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # noinspection PyUnboundLocalVariable
    return clusters


# k_means(
#     data=df[["glucose", "bmi", "age"]],
#     k=2, metric=euclidean_distance,
#     max_iterations=1
# )


def k_nearest(classified: DataFrame, labels: Series, unclassified: DataFrame, k: int,
              metric: Callable[[DataFrame, Series], Series]) -> DataFrame:
    predictions = DataFrame(0, index=unclassified.index, columns=["prediction"])

    for point_id, point in unclassified.iterrows():
        distances = metric(classified, point)
        nearest = distances.nsmallest(k).index
        voting = labels.loc[nearest].mode()

        predictions.loc[point_id, "prediction"] = voting[0]

    return predictions

# k_nearest(
#     classified=df[["glucose", "bmi", "age"]],
#     labels=df["diabetes"],
#     unclassified=df.loc[1:2][["glucose", "bmi", "age"]],
#     k=2, metric=euclidean_distance
# )
