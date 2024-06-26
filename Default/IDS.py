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

quant_dataframe = QuantitativeDataFrame(df)

ids = IDS(algorithm="SLS")
ids.fit(quant_dataframe=quant_dataframe, class_association_rules=cars, lambda_array=lambda_array)

acc = ids.score(quant_dataframe)
print(acc)

ids.score_interpretability_metrics(quant_dataframe)
