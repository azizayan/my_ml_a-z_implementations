import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0).fit(X_train,y_train)

pred_first_customer = classifier.predict((X_test[7].reshape(1,-1)))
pred_first_customer_proba = classifier.predict_proba((X_test[7].reshape(1,-1)))

#if we have the values of the 1 sample we want to predict we can also do:
classifier.predict(sc.transform([[30,87000]]))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



print("Confusion Matrix")
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(accuracy_score(y_test,y_pred))



