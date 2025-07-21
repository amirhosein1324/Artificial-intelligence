import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

base = "/kaggle/input/rice-image-dataset/Rice_Image_Dataset"

from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rescale = 1/255 , validation_split = 0.2)

train = gen.flow_from_directory(base , subset = "training" , target_size=(256,256))

test = gen.flow_from_directory(base , subset = "validation" , target_size = (256,256))

train

model = keras.Sequential([
    layers.Input(shape = (256,256,3)),
    layers.Conv2D(8, kernel_size = (4,4), strides = (1,1),padding = "SAME" , activation = "relu" ),
    layers.MaxPool2D(pool_size = (8,8), strides = (8,8),padding = "SAME"),

    layers.Conv2D(16 , kernel_size = (2,2), strides = (1,1),padding = "SAME" , activation = "relu" ),
    layers.MaxPool2D(pool_size = (4,4), strides = (4,4),padding = "SAME"),
    layers.Flatten(),
    layers.Dense( 5, activation = "softmax" )
])

model.compile (optimizer = "adam" , loss = "categorical_crossentropy" , metrics = ["accuracy"])

model.summary()

history = model.fit(train , epochs = 10)

