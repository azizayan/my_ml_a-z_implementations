import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model,model_selection


X,y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(X[0])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.25)

regressor = linear_model.LinearRegression()
regressor.fit(X,y)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))