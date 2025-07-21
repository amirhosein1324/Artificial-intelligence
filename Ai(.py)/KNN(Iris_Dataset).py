import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("/content/IRIS (1).csv")
df.dropna()
df.head()

df["species"].unique()

X = df.iloc[:, 0:4].values
lc = LabelEncoder()
df["species"] = lc.fit_transform(df["species"])
Y = df["species"].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
neigh

y_hat = neigh.predict(x_test)
y_hat

# metrics.accuracy_score(y_train, neigh.predict(x_train))
metrics.accuracy_score(y_test, y_hat)

