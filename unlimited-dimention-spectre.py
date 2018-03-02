import pickle

from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re

def custom_objective(y_true, y_pred):
  mse = K.mean(K.square(y_true-y_pred), axis=-1)

  y_true_clip = K.clip(y_true, K.epsilon(), 1)
  y_pred_clip = K.clip(y_pred, K.epsilon(), 1)
  kullback_leibler = K.sum(y_true_clip * K.log(y_true_clip / y_pred_clip), axis=-1)
  return mse + kullback_leibler / 1000.0

buff = None
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )


inputs      = Input(shape=(20,))
x           = Dense(5000, activation='relu')(inputs)
x           = Dense(5000, activation='relu')(x)
x           = Dense(5000, activation='relu')(x)
x           = Dense(5000, activation='relu')(x)
x           = Dense(66, activation='relu')(x)

model = Model(inputs, x)
model.compile(optimizer=Adam(lr=0.0001), loss=custom_objective)

if '--train' in sys.argv:
  Xs, Ys, Xst, Yst = pickle.loads( open('dataset.pkl', 'rb').read() )
  
  init  = 0.00001
  decay = 0.01
  for i in range(100):
    lr =  init*(1 - i*decay)
    print(f'{lr:0.9f}')
    model.optimizer = Adam(lr=lr)
    model.fit(Xs, Ys, epochs=1, validation_data=(Xst, Yst), callbacks=[batch_callback])
    
    model.save_weights(f'models/valloss={buff["val_loss"]:.09f}_loss={buff["loss"]:.09f}_{i:09d}.h5')
