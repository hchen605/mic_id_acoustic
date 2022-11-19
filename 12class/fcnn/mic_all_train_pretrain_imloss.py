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
classes_3 = ['C','D','M','P']
classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M2','M3','P1','P2','P3','P4','P5','P6']
genders = ['full', 'female', 'male']

# -

gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12


train_csv = '../data/test_full_mobile_clo_4th_sp1.csv'
dev_csv = '../data/test_full_mobile_clo_4th_sp1.csv'
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

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]

#np.save('../data/ood/y_test.npy',y_train)

y_dev = [cls2label[y] for y in y_dev]
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 64
epochs = args.eps

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model = model_xvector(num_classes)
weights_path = 'weight/weight_full_mobile_limit400_seed0_pre_sp1/'+ experiments + "best.hdf5"
model.load_weights(weights_path)

softmax_out = model.predict(x_train)
#np.save('../data/ood/pseudo_softmax.npy', softmax_out)
#np.save('../data/ood/pseudo_label.npy', np.argmax(softmax_out,axis=-1))
y_true = np.load('../data/ood/pseudo_update_4.npy')
y_true = keras.utils.to_categorical(y_true, num_classes=num_classes)

def lm_loss(y_true, y_pred):
    epsilon = 1e-5
    entropy = -y_pred * tensorflow.math.log(y_pred + epsilon)
    entropy = tensorflow.math.reduce_sum(entropy, axis=1)

    msoftmax = tensorflow.math.reduce_mean(y_pred, axis=0)
    gentropy_loss = tensorflow.math.reduce_sum(-msoftmax * tensorflow.math.log(msoftmax + epsilon))
    entropy -= gentropy_loss

    cce = keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    loss += entropy

    return loss

model.compile(loss=lm_loss,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=[lm_loss])


model.summary()

# Checkpoints
if not os.path.exists('weight/weight_full_mobile_limit{}_seed{}_pre_imloss/'.format(args.limit, args.seed)+experiments):
    os.makedirs('weight/weight_full_mobile_limit{}_seed{}_pre_imloss/'.format(args.limit, args.seed)+experiments)

#save_path = "weight/weight_full_mobile_limit{}_seed{}/".format(args.limit, args.seed)+ experiments + "{epoch:02d}-{val_accuracy:.4f}.hdf5"
save_path = "weight/weight_full_mobile_limit{}_seed{}_pre_imloss/".format(args.limit, args.seed)+ experiments + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [checkpoint]


# Training
exp_history = model.fit(x_train, y_true, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. loss: ", max(exp_history.history['val_loss']), " At Epoch of ", np.argmin(exp_history.history['val_loss'])+1)

# +

