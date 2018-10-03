from __future__ import print_function
#from keras.preprocessing import sequence
import numpy as np
import tensorflow as tf
from keras import backend as K
import keras
import random
import cPickle
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as auprc
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Embedding, Masking, GRU
from keras.layers.core import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

timestamp = 200
num_features = 12
batch = 2048
iter = 0

print('batch size', batch)

x = np.load('input_for_notes_interpolated.npy')
y = np.load('mortality_labels.npy')
notes = np.load('label_notes.npy')

#Dividing into train and test data
ind1 = int(0.6*len(x))
ind2 = int(0.8*len(x))
x_train = x[:ind1]
y_train = y[:ind1]
notes_train = notes[:ind1]
x_val = x[ind1:ind2]
y_val = y[ind1:ind2]
notes_val = notes[ind1:ind2]
x_test = x[ind2:]
y_test = y[ind2:]
notes_test = notes[ind2:]
x = []
y = []
notes = []

print('Build model...')
main_input = Input(shape=(3*num_features, 192), name='main_input')
x = Permute((2, 1))(main_input)
x = GRU(100)(x)
aux_output = Dense(1, activation='sigmoid', name='aux_output')(x)
main_output = Dense(1409, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input], outputs=[main_output, aux_output])


model.load_weights('predict_notes_1.h5')
model.compile(optimizer='adam',
              loss={'main_output': 'binary_crossentropy', 'aux_output':'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 1.}, metrics={'main_output':'accuracy', 'aux_output':'accuracy'})

model.summary()
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0)
callbacks_list = [earlystop]

print('Train...')
model.fit({'main_input': x_train},
          {'main_output': notes_train, 'aux_output': y_train},
          batch_size = batch, callbacks = callbacks_list, epochs=iter, validation_data=(x_val, [notes_val, y_val]))

print('Save model....')
model.save_weights('predict_notes_1.h5')
y_pred = model.predict(x_test, batch_size=batch)
preds = y_pred[0]
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

print('f1_score', f1_score(notes_test, preds, average='micro'))
print('precision_score', precision_score(notes_test, preds, average='micro'))
print('recall_score', recall_score(notes_test, preds, average='micro'))
print(model.evaluate({'main_input': x_test},{'main_output': notes_test, 'aux_output': y_test}, batch_size = batch))



