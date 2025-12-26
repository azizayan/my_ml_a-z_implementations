import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets,linear_model,model_selection


dataset = datasets.load_linnerud()

X = dataset.data
y = dataset.target

print(X.shape)
print(X[0])

print(dataset.feature_names)
print(dataset.target_names)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.25)




regressor = linear_model.LinearRegression()
regressor.fit(X_train,y_train)



y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),3),y_test.reshape(len(y_test),3)),1)) as dataset contains 3 target, predictions will be in shape (n,3)
print(np.concatenate((y_pred, y_test), axis=1))# as dataset is already in true shape, no need to reshape.Full control , requires same number of rows and same dimeonsionality
#np.column_stack((y_pred, y_test))  this automaticlly converts 1D arrays into column vectors, handles shape normalization