import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from random import shuffle

sentences = []
targets = []
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

training_size = train_csv.shape[0]
embedding_dims = 128
learning_rate = 0.0001

for i in range(train_csv.shape[0]):
    keyword = train_csv.iloc[i][1]
    location = train_csv.iloc[i][2]
    text = train_csv.iloc[i][3]
    text = text + str(location) + str(keyword)
    sentences.append(text)

    target = train_csv.iloc[i][4]
    targets.append(target)


train_sentences = sentences[0:training_size]
# test_sentences = sentences[training_size:]
train_labels = targets[0:training_size]
# test_labels = targets[training_size:]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(train_sentences)
training_padded = pad_sequences(training_sequences, padding='post')
maxlen = training_padded.shape[1]
#
# testing_sequences = tokenizer.texts_to_sequences(test_sentences)
# testing_padded = pad_sequences(testing_sequences, padding='post', maxlen=maxlen)


training_padded = np.array(training_padded)
train_labels = np.array(train_labels)
# testing_padded = np.array(testing_padded)
# test_labels = np.array(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index), embedding_dims, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer =tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

num_epochs = 300
history = model.fit(training_padded, train_labels, epochs=num_epochs, verbose=2)


test_sentences = []
for i in range(test_csv.shape[0]):
    keyword = train_csv.iloc[i][1]
    location = train_csv.iloc[i][2]
    text = train_csv.iloc[i][3]
    text = text + str(location) + str(keyword)
    test_sentences.append(text)



sequences = tokenizer.texts_to_sequences(test_sentences)
padded = pad_sequences(sequences,maxlen=maxlen, padding='post')

predictions = model.predict(padded)
predictions = pd.DataFrame(predictions).round(1)
predictions = predictions.astype(int)

my_submission = pd.DataFrame()
my_submission['id'] = test_csv['id']
my_submission['target'] = predictions

my_submission.to_csv('my_submission.csv', index=False)

