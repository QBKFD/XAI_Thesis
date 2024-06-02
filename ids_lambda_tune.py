import pandas as pd
import io
import requests

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent

from pyarc.qcba.data_structures import QuantitativeDataFrame


url = "https://raw.githubusercontent.com/jirifilip/pyids/master/data/titanic.csv"
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