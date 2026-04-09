import random

from pandas import DataFrame


def shuffle_dataset(data: DataFrame) -> None:
    for i in range(len(data)):
        random_id = random.randint(0, len(data) - 1)
        data.iloc[i], data.iloc[random_id] = data.iloc[random_id], data.iloc[i]


def split_dataset(data: DataFrame, percentage: float) -> tuple[DataFrame, DataFrame]:
    assert 0 <= percentage <= 1

    take = int(len(data) * percentage)
    return data[:take], data[take:]
