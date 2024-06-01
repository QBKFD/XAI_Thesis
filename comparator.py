import pandas as pd
import numpy as np
import io
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from ruleopt import *
from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini
from ruleopt.solver import ORToolsSolver
from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyarc.qcba.data_structures import QuantitativeDataFrame



url = "https://raw.githubusercontent.com/QBKFD/XAI_Thesis/main/Data/COMPAS_binary.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))



X = df.iloc[:, :-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_corels, y_train_corels = X_train, y_train
X_test_corels, y_test_corels = X_test, y_test

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y,y_pred_corels)
    precision = precision_score(y, y_pred_corels, average='weighted')
    recall = recall_score(y, y_pred_corels, average='weighted')
    f1 = f1_score(y, y_pred_corels, average='weighted')
    return accuracy, precision, recall, f1

################################################################################################################################################################

#CORELS model
c = CorelsClassifier(n_iter=10000)

# Fit the model. Features is a list of the feature names

c.fit(X_train, y_train, features=features, prediction_name=prediction)

#Prediction
y_pred_corels = c.predict(X_test)

accuracy_corels, precision_corels, recall_corels, f1_corels = evaluate_model(y_test, y_pred_corels)

################################################################################################################################################################

tree_parameters = {"max_depth": 3, "class_weight": "balanced"}
solver = ORToolsSolver()
rule_cost = Gini()
random_state = 42

# Initialize the RUGClassifier with specific parameters
rug = RUGClassifier(
    solver=solver,
    random_state=random_state,
    max_rmp_calls=20,
    rule_cost=rule_cost,
    **tree_parameters,
)


# Fit the RUGClassifier to the training data
rug.fit(X_train, y_train)

# Predict the labels of the testing set
y_pred_opt = rug.predict(X_test)

accuracy_opt, precision_opt, recall_opt, f1_opt = evaluate_model(y_test, y_pred_opt)


################################################################################################################################################################
#IDS Model

quant_df = QuantitativeDataFrame(df)
cars = mine_CARs(df, 20)
accuracy_ids = ids.score(quant_dataframe)

################################################################################################################################################################

comparison_table = pd.DataFrame({
    'Model': ['CORELS', 'RuleOpt', 'IDS'],
    'Accuracy': [accuracy_corels, accuracy_opt, accuracy_ids],
    'Precision': [precision_corels, precision_opt, np.nan],  # Exclude IDS from precision
    'F1 Score': [f1_corels, f1_opt, np.nan]  # Exclude IDS from F1 Score
})
print(comparison_table)

################################################################################################################################################################

# Plotting the comparison
metrics = ['Accuracy', 'Precision', 'F1 Score']
colors = ['b', 'g', 'r']
width = 0.2

fig, ax = plt.subplots()

# CORELS metrics
corels_metrics = [accuracy_corels, precision_corels, f1_corels]
ax.bar(np.arange(len(metrics)) - width, corels_metrics, width, label='CORELS', color='b')

# RuleOpt metrics
rug_metrics = [accuracy_opt, precision_opt, f1_opt]
ax.bar(np.arange(len(metrics)), rug_metrics, width, label='RuleOpt', color='g')

# IDS metrics
ids_metrics = [accuracy_ids, np.nan, np.nan]  # IDS only has accuracy
ax.bar(np.arange(len(metrics)) + width, ids_metrics, width, label='IDS', color='r')

#Labels and titles
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Metrics')
ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metrics)
ax.legend()

plt.show()