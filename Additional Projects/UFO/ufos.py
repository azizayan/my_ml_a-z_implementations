import pandas as pd
import numpy as np

dataset = pd.read_csv('ml_practices/ufos.csv')


smaller_dataset = pd.DataFrame({'Seconds':dataset['duration (seconds)'],'Country': dataset['country'],'Latitude':dataset['latitude'],'Longitude':dataset['longitude'],})

print(smaller_dataset.Country.unique())

smaller_dataset.dropna(inplace=True)