from corels import *

# Load the dataset
X, y, _, _ = load_from_csv("/content/compas.csv")

# Create the model, with 10000 as the maximum number of iterations
c = CorelsClassifier(n_iter=10000)

# Fit, and score the model on the training set
a = c.fit(X, y).score(X, y)

# Print the model's accuracy on the training set
print(a)
