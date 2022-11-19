import os
import sys
sys.path.append("..")
import argparse

import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam

from utils import *
from funcs import *

from ts_dataloader import *
from models.small_fcnn_att import model_fcnn
from models.xvector import model_xvector

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import random


parser = argparse.ArgumentParser()
parser.add_argument("--gsc", type=int, default=0, help="training gsc")
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
parser.add_argument("--target", type=int, default=0, help="mic (0), room(1)")
parser.add_argument("--nclass", type=int, default=0, help="3class (0), 12class(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")

args = parser.parse_args()

#tensorflow.reset_default_graph()
os.environ['PYTHONHASHSEED']= '0'#str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras import backend as K
config = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
#tensorflow.keras.utils.set_random_seed(args.seed)
sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=config)
K.set_session(sess)

# +
classes_3 = ['sma','med','lar']
classes_12 = ['sma1','sma1','med1','med2','med3','lar1','lar2','lar2']





train_csv = '../data/train_room.csv'
dev_csv = '../data/dev_room.csv'
#test_csv = '../data/test_dimension.csv'




print('loading microphone data')
train = load_data(train_csv)
dev = load_data(dev_csv)
#test = load_data(test_csv)

if args.limit < 300:
    train = split(train, args.limit)

print ("=== Number of training data: {}".format(len(train)))
#print ("=== Number of test data: {}".format(len(test)))


x_train, y_train_3, y_train_12 = list(zip(*train))
x_dev, y_dev_3, y_dev_12 = list(zip(*dev))
#x_test, y_test_3, y_test_12 = list(zip(*test))
x_train = np.array(x_train)
x_dev = np.array(x_dev)
#x_test = np.array(x_test)



if args.nclass == 0:
    y_train = y_train_3
    y_dev = y_dev_3
    #y_test = y_test_3
    classes = classes_3
    experiments = '3class/'
else:
    y_train = y_train_12
    y_dev = y_dev_12
    #y_test = y_test_12
    classes = classes_12
    experiments = '12class/'

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]
y_dev = [cls2label[y] for y in y_dev]
#y_test = [cls2label[y] for y in y_test]
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model = model_xvector(num_classes)

#weights_path = 'weight_limit50_dim_noise_m/weight.hdf5'
#model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


model.summary()


# Checkpoints
if not os.path.exists('weight/weight_room_limit{}_x/'.format(args.limit)+experiments):
    os.makedirs('weight/weight_room_limit{}_x/'.format(args.limit)+experiments)

save_path = "weight/weight_room_limit{}_x/".format(args.limit)+ experiments + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks = [checkpoint]




# Training
exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. Acc: ", max(exp_history.history['val_accuracy']), " At Epoch of ", np.argmax(exp_history.history['val_accuracy'])+1)

