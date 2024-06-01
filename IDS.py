import pandas as pd
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS

from pyarc.qcba.data_structures import QuantitativeDataFrame
import io
import requests

url = "https://raw.githubusercontent.com/QBKFD/XAI_Thesis/main/Data/breast-cancer-wisconsin%20(1).csv?token=GHSAT0AAAAAACR3CZDHPD4SBFMX6BOBKMIKZSMTKPQ"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
#df = pd.read_csv("https://github.com/tabearoeber/BAN-thesis-2024/blob/main/datasets/binary/DIABETES_PIMA_binary.csv")
cars = mine_CARs(df, rule_cutoff=50)
lambda_array = [1, 1, 1, 1, 1, 1, 1]


#First divide the dataset for train and test partition






quant_dataframe = QuantitativeDataFrame(df)

ids = IDS(algorithm="SLS")
ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

acc = ids.score(quant_dataframe)
print(acc)

ids.score_interpretability_metrics(quant_dataframe)

import pandas as pd
import io
import requests

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent

from pyarc.qcba.data_structures import QuantitativeDataFrame


url = "https://raw.githubusercontent.com/QBKFD/XAI_Thesis/main/Data/breast-cancer-wisconsin%20(1).csv?token=GHSAT0AAAAAACR3CZDHPD4SBFMX6BOBKMIKZSMTKPQ"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
quant_df = QuantitativeDataFrame(df)
cars = mine_CARs(df, 20)


def fmax(lambda_dict):
    print(lambda_dict)
    ids = IDS(algorithm="SLS")
    ids.fit(class_association_rules=cars, quant_dataframe=quant_df, lambda_array=list(lambda_dict.values()))
    auc = ids.score_auc(quant_df)
    print(auc)
    return auc



coord_asc = CoordinateAscent(
    func=fmax,
    func_args_ranges=dict(
        l1=(1, 1000),
        l2=(1, 1000),
        l3=(1, 1000),
        l4=(1, 1000),
        l5=(1, 1000),
        l6=(1, 1000),
        l7=(1, 1000)
    ),
    ternary_search_precision=50,
    max_iterations=3
)

best_lambdas = coord_asc.fit()
