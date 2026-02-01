import pandas as pd
import numpy as np
import pickle


dataset = pd.read_csv('ml_practices/ufos.csv')


ufos = pd.DataFrame({'Seconds':dataset['duration (seconds)'],'Country': dataset['country'],'Latitude':dataset['latitude'],'Longitude':dataset['longitude'],})

print(ufos.Country.unique())

ufos.dropna(inplace=True)

ufos.info()

from sklearn.preprocessing import LabelEncoder


ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

print(ufos.head())


from sklearn.model_selection import train_test_split

X = ufos[['Seconds','Latitude','Longitude']]
y= ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 0)

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))

model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model= pickle.load(open('ufo-model.pkl','rb'))

