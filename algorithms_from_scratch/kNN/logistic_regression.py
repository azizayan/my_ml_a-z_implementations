import numpy as np
import math
import csv
from pathlib import Path



class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):

        self.lr = learning_rate
        self.n_iter= n_iterations
        self.weights = None
        self.bias = None


    def _sigmoid(self,z):
        exponent = np.exp(-z)#forgot minus here
        divider = exponent + 1
        result = 1/ divider

        return result

    def _compute_loss(self, y_true, y_pred):
        first_component = y_true * np.log(y_pred)
        second_component = (1-y_true)* np.log(1-y_pred)

        result = -np.mean(first_component+second_component)


        return result


    def fit(self, X,y):
        

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)#how to create a box of zeros for bunch of data
        self.bias = 0

        for _  in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self._sigmoid(linear_model)


            dw = (np.dot(X.T, y_pred - y))/n_samples
            db = (np.sum(y_pred - y))/n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
       
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            return [1 if i > 0.5 else 0 for i in y_pred]



def load_csv(filename: Path):
    dataset = []
    with open(filename,'r') as file:

        csv_reader = csv.reader(file)

        next(csv_reader)

        for row in csv_reader:

            if not row:
                continue

            cleaned_row = []

            for i in range(4):
                cleaned_row.append(float(row[i]))

            cleaned_row.append(row[4])

            dataset.append(cleaned_row)

    return dataset



X_raw = [
    [1.0, 1.5],  # Student 1: Lazy (Fail)
    [2.0, 1.0],  # Student 2: Lazy (Fail)
    [1.5, 2.5],  # Student 3: Lazy (Fail)
    [3.0, 1.0],  # Student 4: Lazy (Fail)
    [2.0, 2.0],  # Student 5: Lazy (Fail)
    
    [8.0, 8.5],  # Student 6: Hardworker (Pass)
    [9.0, 9.0],  # Student 7: Hardworker (Pass)
    [7.5, 8.0],  # Student 8: Hardworker (Pass)
    [9.0, 7.5],  # Student 9: Hardworker (Pass)
    [8.0, 9.0]   # Student 10: Hardworker (Pass)
]

# y = [0 = Fail, 1 = Pass]
y_raw = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 1. Prepare Data
X = np.array(X_raw)
y = np.array(y_raw)

# 2. Train
# Use a high learning rate because this data is very easy
model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
model.fit(X, y)

# 3. Test (Predict on the same data to verify)
predictions = model.predict(X)

accuracy = np.mean(predictions == y)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

print(f"True Labels: {y}")
print(f"Predictions: {predictions}")


print(f"Final Weights: {model.weights}")
print(f"Final Bias: {model.bias}")
        