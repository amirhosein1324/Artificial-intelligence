import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("/content/iris.csv")
df.dropna()
df.head()

lc = LabelEncoder()
df["species"] = lc.fit_transform(df["species"])
df.head()

X = df.iloc[:, 0:4]
Y = df.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras.Sequential([
    layers.Dense(256, input_shape=[4], activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20)

