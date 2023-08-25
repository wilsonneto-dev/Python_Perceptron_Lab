import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_func(linear_output)
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._unit_step_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


# Sample data
X = np.array([[3], [4.5], [6.5], [7.5], [8], [9], [5.5], [6], [7], [9.5]])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])

# Train the perceptron
p = Perceptron(learning_rate=0.01, n_iterations=1000)
p.fit(X, y)

# Test data
test_scores = np.array([[6.8], [7.2], [8.5], [5.4]])
predictions = p.predict(test_scores)

for score, pred in zip(test_scores, predictions):
    status = "approved" if pred == 1 else "not approved"
    print(f"Score {score[0]:.1f} is {status}")
