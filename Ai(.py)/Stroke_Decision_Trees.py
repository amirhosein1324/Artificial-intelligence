import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv("/content/healthcare-dataset-stroke-data.csv")
df = df.dropna()
df.head()

lc = LabelEncoder()
df["gender"] = lc.fit_transform(df["gender"])
df["ever_married"] = lc.fit_transform(df["ever_married"])
df["work_type"] = lc.fit_transform(df["work_type"])
df["Residence_type"] = lc.fit_transform(df["Residence_type"])
df["smoking_status"] = lc.fit_transform(df["smoking_status"])
X = df[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]]
Y = df[["stroke"]]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

d_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
d_tree.fit(x_train, y_train)

y_hat = d_tree.predict(x_test)
y_hat

metrics.accuracy_score(y_test, y_hat)

