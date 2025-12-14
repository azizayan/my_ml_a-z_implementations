import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv('Social_Network_Ads.csv')
 

from sklearn.model_selection import train_test_split


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

from sklearn import svm

classifier = svm.SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


print(y_pred)

print(classifier.support_vectors_)


print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
