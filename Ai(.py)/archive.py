import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

df = pd.read_csv("/content/spam.csv" , encoding = "utf-8" ,
                encoding_errors = "ignore" )

df.head()

df.replace({"ham":0 , "spam":1} , inplace = True)

Y = df.v1.tolist()

X = df.v2.tolist()

len(X)

df.head()

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index

vocab_size=5572
embedding_dim=16
max_length=max([len(item.split()) for item in X])
trunc_type='post'
padding_type='post'
oov_tok="<oov>"
training_size=2000

max([len(item.split()) for item in X])

train_x = X[0:training_size]
test_x = X[training_size:]
train_y = Y[0:training_size]
test_y = Y[training_size:]

tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)

word_index=tokenizer.word_index

training_sequences=tokenizer.texts_to_sequences(train_x)
training_padded=pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(test_x)
testing_padded=pad_sequences(testing_sequences, maxlen=max_length,padding=padding_type, truncating=trunc_type )

training_padded=np.array(training_padded)
training_labels=np.array(train_y)
testing_padded=np.array(testing_padded)
testing_labels=np.array(test_y)

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs=50
history=model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

