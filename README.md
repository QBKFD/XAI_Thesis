# XAI Thesis Project
# Rule-Based Methods Comparison Using Mathematical Optimization

This repository contains the implementation and comparison of rule-based methods using mathematical optimization techniques. The primary focus is on comparing the following algorithms:
- RuleOPT
- Interpretable Decision Sets (IDS) algorithm
- CORELS (Certifiably Optimal Rule Lists)
- MDL(Minimum Decision Length) algorithm

## Project Description

The objective of this project is to evaluate and compare the performance of different rule-based algorithms in terms of their interpretability and accuracy. The methods compared in this study leverage mathematical optimization to generate decision sets or lists that are easy to understand and interpret.

## Algorithms Compared

### RuleOPT
RuleOPT is a hierarchical rule-based algorithm that prioritizes rules using weights to manage complex datasets. It focuses on optimizing the overall decision set for both accuracy and interpretability.

### Interpretable Decision Sets (IDS)
IDS, developed by Lakkaraju et al. (2020), uses a collection of unordered classification rules. Each rule operates independently, making it easier to interpret and understand. The IDS framework aims to create decision sets that are near-optimal in size and interpretability.

### CORELS (Certifiably Optimal RulE ListS)
CORELS is an algorithm that generates rule lists with provable optimality. It focuses on producing rule lists that are both accurate and easy to interpret, ensuring that the generated models are transparent.
### MDL (Minimum Decision Length)
MDL is a principle used in model selection that aims to minimize the total length of the description of the model and the data it encodes. This approach helps in finding models that are both simple and effective in explaining the data.

## Baselines

The baseline algorithms tested are:

- **Random Forest**: A robust ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.
- **C4.5 Decision Tree**: An extension of the ID3 algorithm that handles both categorical and numerical data, and uses entropy to create decision trees.
- **CART (Classification and Regression Trees)**: A decision tree algorithm that can be used for both classification and regression tasks.


## Hypertuning

### GridSearchCV:
- RuleOpt
### Internal lambda parameters
- IDS

## Authors and References

### RuleOPT
- Lumadjenga, Röbera, Akyüz, Birbila
- [https://github.com/sametcopur/ruleopt](#)
- [https://arxiv.org/abs/2104.10751](#)

### Interpretable Decision Sets (IDS)
- Lakkaraju, Bach, Leskovec
- [https://github.com/jirifilip/pyIDS](#)
- [https://cs.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf](#)

### CORELS
- [https://github.com/corels/corels](#)

### MDL
- Proença, van Leeuwen
-[https://github.com/HMProenca/MDLRuleLists](#)
-[https://arxiv.org/abs/1905.00328](#)

## Installation

To run the comparisons, ensure you have the following dependencies installed:

- Python 3.10.14
- pandas
- numpy
- scikit-learn

You can install the required packages using:

```bash
pip install -r requirements.txt
