import pandas as pd

marketplace = pd.read_csv("../../data/stragan.csv")

print(marketplace.head())

"""
   swieze  mrozone  ostre  slodkie  ...  tropikalne  lisciaste  bulwowe      nazwa
0     1.0      0.0    0.0      0.7  ...           0        1.0      0.0     jablko
1     0.0      0.8    0.4      0.0  ...           0        0.0      1.0  marchewka
2     0.0      1.0    0.0      1.0  ...           0        0.5      0.5  truskawki
3     0.9      0.0    0.0      0.3  ...           0        0.0      1.0     ogorek
4     0.2      0.0    0.0      0.0  ...           1        1.0      0.0    cytryna
"""

decisive_class = "nazwa"

customer_preferences = {
    "A": ["swieze", "ostre", "czerwone"],
    "B": ["mrozone", "zielone", "slodkie", "lisciaste"],
    "C": ["swieze", "zielone", "czerwone", "slodkie"]
}

for customer_name, preferred_features in customer_preferences.items():
    print(f"Klient {customer_name}")

    soft_set = marketplace[[*preferred_features, decisive_class]]
    soft_set["choice"] = soft_set[preferred_features].sum(axis=1)

    print(soft_set)

    choice = soft_set[soft_set["choice"] == soft_set["choice"].max()]

    print(f"Wybierze {", ".join(choice[decisive_class].values)}")
    print()

"""
Klient A
   swieze  ostre  czerwone       nazwa  choice
0     1.0    0.0       0.5      jablko     1.5
1     0.0    0.4       0.8   marchewka     1.2
2     0.0    0.0       1.0   truskawki     1.0
3     0.9    0.0       0.0      ogorek     0.9
4     0.2    0.0       0.3     cytryna     0.5
5     0.9    0.0       1.0     pomidor     1.9
6     0.5    0.0       0.6      banany     1.1
7     0.5    0.0       0.6  pomarancza     1.1
8     0.0    1.0       0.8     fasolka     1.8
Wybierze pomidor

Klient B
   mrozone  zielone  slodkie  lisciaste       nazwa  choice
0      0.0      0.5      0.7        1.0      jablko     2.2
1      0.8      0.2      0.0        0.0   marchewka     1.0
2      1.0      0.0      1.0        0.5   truskawki     2.5
3      0.0      1.0      0.3        0.0      ogorek     1.3
4      0.0      0.0      0.0        1.0     cytryna     1.0
5      0.0      0.0      0.8        0.6     pomidor     1.4
6      0.0      0.2      0.3        1.0      banany     1.5
7      0.0      0.0      0.5        1.0  pomarancza     1.5
8      0.7      0.7      0.0        0.0     fasolka     1.4
Wybierze truskawki

Klient C
   swieze  zielone  czerwone  slodkie       nazwa  choice
0     1.0      0.5       0.5      0.7      jablko     2.7
1     0.0      0.2       0.8      0.0   marchewka     1.0
2     0.0      0.0       1.0      1.0   truskawki     2.0
3     0.9      1.0       0.0      0.3      ogorek     2.2
4     0.2      0.0       0.3      0.0     cytryna     0.5
5     0.9      0.0       1.0      0.8     pomidor     2.7
6     0.5      0.2       0.6      0.3      banany     1.6
7     0.5      0.0       0.6      0.5  pomarancza     1.6
8     0.0      0.7       0.8      0.0     fasolka     1.5
Wybierze jablko, pomidor
"""
