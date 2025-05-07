import numpy as np

# Step 1: Linear function f(x) = w*x + b
def linear_forward(X, weights, bias):
    return np.dot(X, weights) + bias

# Step 2: Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Example inputs
X = np.array([[2], [4], [6]])    # Feature values
weights = np.array([0.5])        # m
bias = -4                        # b

# Linear forward pass
z = linear_forward(X, weights, bias)

# Logistic (sigmoid) activation
y_hat = sigmoid(z)

# Output
print("Linear output (z):\n", z)
print("Logistic output (sigmoid):\n", y_hat)
