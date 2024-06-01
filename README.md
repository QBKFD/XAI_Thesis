
# XAI_Thesis
# Rule-Based Method Comparison Using Mathematical Optimization

This repository contains the implementation and comparison of rule-based methods using mathematical optimization techniques. The primary focus is on comparing the following algorithms:
- RuleOPT
- Interpretable Decision Sets (IDS)
- CORELS (Certifiably Optimal RulE ListS)

## Project Description

The objective of this project is to evaluate and compare the performance of different rule-based algorithms in terms of their interpretability and accuracy. The methods compared in this study leverage mathematical optimization to generate decision sets or lists that are easy to understand and interpret.

## Algorithms Compared

### RuleOPT
RuleOPT is a hierarchical rule-based algorithm that prioritizes rules using weights to manage complex datasets. It focuses on optimizing the overall decision set for both accuracy and interpretability.

### Interpretable Decision Sets (IDS)
IDS, developed by Lakkaraju et al. (2020), uses a collection of unordered classification rules. Each rule operates independently, making it easier to interpret and understand. The IDS framework aims to create decision sets that are near-optimal in size and interpretability.

### CORELS (Certifiably Optimal RulE ListS)
CORELS is an algorithm that generates rule lists with provable optimality. It focuses on producing rule lists that are both accurate and easy to interpret, ensuring that the generated models are transparent and comprehensible.

## Authors and References

### RuleOPT
- Lumadjeng et al.,
- [https://github.com/sametcopur/ruleopt](#)
- [https://arxiv.org/abs/2104.10751](#)](#)

### Interpretable Decision Sets (IDS)
- Lakkaraju et al. (2020)
- [https://github.com/jirifilip/pyIDS](#)
- [https://cs.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf](#)

### CORELS
- [https://github.com/corels/corels](#)

## Installation

To run the comparisons, ensure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- scikit-learn

You can install the required packages using:

```bash
pip install -r requirements.txt

