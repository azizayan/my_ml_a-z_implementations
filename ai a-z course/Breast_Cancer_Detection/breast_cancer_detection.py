import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#actully there is no need the ucimlrepo import here, for future use 
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_original.variables) 
#code until here is for future use


print("---------------- ACTUAL CODE ----------------")

# data (as pandas dataframes) 
dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:,1:-1].values#sample code number at index 0, ignore it
y = dataset.iloc[:,-1].values#class(2 for beign 4 for malign)
  
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X = X_train, y= y_train,cv=10)

print(accuracies.mean())
print(accuracies.std())


