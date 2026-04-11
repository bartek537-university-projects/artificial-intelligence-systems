import random
from typing import Callable

import pandas as pd
from pandas import DataFrame, Series

iris = pd.read_csv("../../data/iris.csv")

decisive_class = "variety"
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

# Przed przystąpieniem do dalszej analizy, zamieńmy najpierw tekstowe opisy gatunków na liczby.
forward_variety_mapping = iris[decisive_class].mode()
reverse_variety_mapping = forward_variety_mapping.reset_index().set_index(decisive_class)["index"]
iris[decisive_class] = iris[decisive_class].map(reverse_variety_mapping)


def shuffle_dataset(data: DataFrame) -> None:
    for i in range(len(data)):
        random_id = random.randint(0, len(data) - 1)
        data.iloc[i], data.iloc[random_id] = data.iloc[random_id], data.iloc[i]


def split_dataset(data: DataFrame, percentage: float) -> tuple[DataFrame, DataFrame]:
    assert 0 <= percentage <= 1

    take = int(len(data) * percentage)
    return data[:take], data[take:]


# Podzielmy również zbiór na dane treningowe i walidacyjne.
shuffle_dataset(iris)
train_data, validation_data = split_dataset(iris, 0.7)

X_train, y_train = train_data[features], train_data[decisive_class]
X_validation, y_validation = validation_data[features], validation_data[decisive_class]

# Przed użyciem algorytmu KNN musimy także znormalizować dane. Można to zrobić funkcją min-max.
lower_bound, upper_bound = X_train.min(), X_train.max()

X_train_norm = (X_train - lower_bound) / (upper_bound - lower_bound)
X_validation_norm = (X_validation - lower_bound) / (upper_bound - lower_bound)


def manhattan_distance(data: DataFrame, centroid: Series) -> Series:
    return (data - centroid).abs().sum(axis=1)


def k_nearest(classified: DataFrame, labels: Series, unclassified: DataFrame, k: int,
              metric: Callable[[DataFrame, Series], Series]) -> DataFrame:
    predictions = DataFrame(0, index=unclassified.index, columns=["prediction"])

    for point_id, point in unclassified.iterrows():
        distances = metric(classified, point)
        nearest = distances.nsmallest(k).index
        voting = labels.loc[nearest].mode()

        predictions.loc[point_id, "prediction"] = voting[0]

    return predictions


# Przetestujmy teraz algorytm dla metryki Manhattan, dla k = 2, 3, 4.
for k in [2, 3, 4]:
    y_predicted = k_nearest(X_train_norm, y_train, X_validation_norm, k, metric=manhattan_distance)["prediction"]

    valid_predictions = (y_predicted == y_validation).sum()
    total_predictions = len(y_predicted)

    accuracy = valid_predictions / total_predictions

    print(f"Dokładność dla k={k} wynosi {accuracy * 100:.2f}%")

# Dokładność dla k=2 wynosi 95.56%
# Dokładność dla k=3 wynosi 97.78%
# Dokładność dla k=4 wynosi 97.78%
