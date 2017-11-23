import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy import stats
import numpy as np
import pickle
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# cleaning data
df = pd.read_csv('2017-11-21_230246.csv')
df = df.drop('state', 1)
le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
city_zip_df = df[['city', 'zip']]
df = df.drop(['city', 'zip'], 1)

df = df.dropna()
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

df = pd.concat([df, city_zip_df], 1)
df = df.dropna()

X = df.drop('price', 1)
y = df['price']

# training model
lr = LinearRegression()

# feature selection
three_feature_df = SelectKBest(chi2, k=3).fit_transform(X, y)
two_feature_df = SelectKBest(chi2, k=2).fit_transform(X, y)
one_feature_df = SelectKBest(chi2, k=1).fit_transform(X, y)
for x, name in zip([X, three_feature_df, two_feature_df, one_feature_df], ['all', 'one', 'two', 'three']):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    lr.fit(X_train, y_train)
    [print(x, y) for x, y in zip(lr.predict(X_test), y_test)]


# serializing file
filename = 'lr_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(lr, file)
