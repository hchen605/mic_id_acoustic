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
from models.attRNN import AttRNN, AttRNN_pre

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


parser = argparse.ArgumentParser()
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
parser.add_argument("--nclass", type=int, default=0, help="3class (0), 12class(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")
args = parser.parse_args()


#tensorflow.reset_default_graph()
os.environ['PYTHONHASHSEED']= str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# +
classes_3 = ['P','X']
#classes_12 = ['P1','P2','P3','P4','X']
classes_12 = ['P3','P4','X']
#classes_12 = ['P1','P2','P3','P4']
genders = ['full', 'female', 'male']

# -

gender = genders[args.gender]




#train_csv = '../data/ood/train_asvspoof_abstention_p.csv'
#dev_csv = '../data/ood/dev_asvspoof_abstention_p.csv'
train_csv = '../data/ood/train_full_mobile_clo_4th_abstention_p_trial_p3p4_dev.csv'
dev_csv = '../data/ood/dev_full_mobile_clo_4th_abstention_p_trial_p3p4_dev.csv'
#test_csv = '../data/test_full_mobile_clo.csv'



print('loading microphone data')
train = load_data(train_csv)
dev = load_data(dev_csv)
#test = load_data(test_csv)

if args.limit < 200:
    train = split_seed(train, args.limit, args.seed)

print ("=== Number of training data: {}".format(len(train)))
#print ("=== Number of test data: {}".format(len(test)))

x_train, y_train_3, y_train_12 = list(zip(*train))
x_dev, y_dev_3, y_dev_12 = list(zip(*dev))
x_train = np.array(x_train)
x_dev = np.array(x_dev)



if args.nclass == 0:
    y_train = y_train_3
    y_dev = y_dev_3
    classes = classes_3
    experiments = '{}/3class/'.format(gender)
else:
    y_train = y_train_12
    y_dev = y_dev_12
    classes = classes_12
    experiments = '{}/12class/'.format(gender)

#print(y_train)
cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]
y_dev = [cls2label[y] for y in y_dev]
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps
input_length = 48000

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model = model_xvector(num_classes)
#model = AttRNN(num_classes, input_length)
#weights_path = '/home/hsinhung/SpeechCmdRecognition/model-attRNN.h5'
#model.load_weights(weights_path)
#model = AttRNN_pre(num_classes, input_length)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


model.summary()

# Checkpoints
if not os.path.exists('weight/weight_asvspoof_abstention_p3p4_id/'):
    os.makedirs('weight/weight_asvspoof_abstention_p3p4_id/')

#save_path = "weight/weight_full_mobile_limit{}_seed{}/".format(args.limit, args.seed)+ experiments + "{epoch:02d}-{val_accuracy:.4f}.hdf5"
save_path = "weight/weight_asvspoof_abstention_p3p4_id/" + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks = [checkpoint]


# Training
exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. Acc: ", max(exp_history.history['val_accuracy']), " At Epoch of ", np.argmax(exp_history.history['val_accuracy'])+1)

# +

