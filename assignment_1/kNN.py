import pandas as pd
import numpy as np
import sklearn

dataframe = pd.read_csv("Smarket.csv")
print(dataframe.head())
print(dataframe.shape)
features = dataframe[['Lag1','Lag2']]
label = dataframe['Direction']
print(features)
print(label)