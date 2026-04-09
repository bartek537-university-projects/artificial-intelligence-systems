import random

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

iris = pd.read_csv("../../data/iris.csv")

# Wyświetlamy krótkie informacje o ilości wierszy, kolumn, typach danych i ewentualnych wartościach pustych.
print(iris.info())
"""
<class 'pandas.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal.length  150 non-null    float64
 1   sepal.width   150 non-null    float64
 2   petal.length  150 non-null    float64
 3   petal.width   150 non-null    float64
 4   variety       150 non-null    str    
dtypes: float64(4), str(1)
memory usage: 6.0 KB
"""
# W naszym zbiorze występuje 150 wierszy i 5 kolumn, w tym 4 są cechami numerycznymi, a 1 etykietą.

# Wyświetlamy kilka pierwszych rekordów.
print(iris.head())
"""
   sepal.length  sepal.width  petal.length  petal.width variety
0           5.1          3.5           1.4          0.2  Setosa
1           4.9          3.0           1.4          0.2  Setosa
2           4.7          3.2           1.3          0.2  Setosa
3           4.6          3.1           1.5          0.2  Setosa
4           5.0          3.6           1.4          0.2  Setosa
"""

features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
decisive_class = "variety"

# Zliczamy wartości w klasie decyzyjnej.
print(iris[decisive_class].value_counts())
"""
variety
Setosa        50
Versicolor    50
Virginica     50
Name: count, dtype: int64
"""
# W zbiorze występuje dokładnie 50 wystąpień każdej wartości, dlatego możemy powiedzieć, że jest on zbalansowany.

# Wygenerujemy także podsumowanie danych numerycznych
# zawierające minimum, maksimum, średnią, odchylenie standardowe oraz kwartyle.
print(iris.describe())
"""
       sepal.length  sepal.width  petal.length  petal.width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
"""
# petal.length ma stosunkowo wysokie odchylenie standardowe, dlatego może dobrze różnicować klasy.

# Aby zrozumieć zależności między danymi, najlepiej będzie wygenerować wykres pairplot
# dla każdej kombinacji kolumn.
sns.pairplot(iris, hue="variety", markers="+")
plt.show()
# Można zaobserwować silne rozdzielenie gatunku Setosa na kilku wykresach, w szczególności na wykresie zależności szerokości i długości płatków.
# Zależność szerokości od długości działki kielicha jest natomiast bardzo rozmyta i klasy się nakładają — nie będzie to dobre do klasyfikacji.

# Możemy przyjrzeć się temu wykresowi bliżej.
sns.scatterplot(iris, x="petal.length", y="petal.width", hue="variety", marker="+")
plt.show()


# Przygotujmy teraz zbiór danych do dalszej analizy,
# tasując zbiór i dzieląc go na zbiór treningowy i testowy.

def shuffle_dataset(dataset: DataFrame) -> None:
    for i in range(len(dataset)):
        random_index = random.randint(0, len(dataset) - 1)
        dataset.iloc[i], dataset.iloc[random_index] = dataset.iloc[random_index], dataset.iloc[i]


def split_dataset(dataset: DataFrame, percentage: float) -> tuple[DataFrame, DataFrame]:
    assert 0 <= percentage <= 1

    take = int(len(dataset) * percentage)
    return dataset[:take], dataset[take:]


shuffle_dataset(iris)
iris_train, iris_test = split_dataset(iris, 0.7)

print(iris_train.head())
print(iris_train.info())
"""
   sepal.length  sepal.width  petal.length  petal.width     variety
0           4.5          2.3           1.3          0.3      Setosa
1           5.8          2.7           5.1          1.9   Virginica
2           4.8          3.0           1.4          0.1      Setosa
3           6.3          3.4           5.6          2.4   Virginica
4           6.8          2.8           4.8          1.4  Versicolor
RangeIndex: 105 entries, 0 to 104
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal.length  105 non-null    float64
 1   sepal.width   105 non-null    float64
 2   petal.length  105 non-null    float64
 3   petal.width   105 non-null    float64
 4   variety       105 non-null    str    
dtypes: float64(4), str(1)
memory usage: 4.2 KB
"""
print(iris_test.head())
print(iris_test.info())
"""
     sepal.length  sepal.width  petal.length  petal.width     variety
105           5.0          3.4           1.6          0.4      Setosa
106           5.5          4.2           1.4          0.2      Setosa
107           5.7          4.4           1.5          0.4      Setosa
108           5.7          2.9           4.2          1.3  Versicolor
109           6.8          3.0           5.5          2.1   Virginica
RangeIndex: 45 entries, 105 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal.length  45 non-null     float64
 1   sepal.width   45 non-null     float64
 2   petal.length  45 non-null     float64
 3   petal.width   45 non-null     float64
 4   variety       45 non-null     str    
dtypes: float64(4), str(1)
memory usage: 1.9 KB
"""
# Jak widać dane zostały odpowiednio wymieszane i podzielone.
