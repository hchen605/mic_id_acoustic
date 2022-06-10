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
from models.small_fcnn_att_rir import model_fcnn_rir, model_fcnn_rir, model_fcnn_mic, model_mic_rir

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


parser = argparse.ArgumentParser()
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
parser.add_argument("--unseen", type=int, default=0, help="seen (0), unseen(1)")
parser.add_argument("--target", type=int, default=0, help="mic (0), room(1)")
parser.add_argument("--nclass", type=int, default=0, help="3class (0), 12class(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
args = parser.parse_args()
'''
#tensorflow.reset_default_graph()
#os.environ['PYTHONHASHSEED']=str(args.seed)
os.environ['PYTHONHASHSEED']= '0'#str(args.seed)
tensorflow.random.set_seed(args.seed)
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
'''
#tensorflow.reset_default_graph()
#os.environ['PYTHONHASHSEED']=str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# +
classes_3 = ['C','D','M']
classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M2','M3']
genders = ['full', 'female', 'male']
classes_room = ['large','medium','small']
class_m = ['mic','dimension']





gender = genders[args.gender]
#target = class_m[args.target]


if args.unseen:
    test_csv = '../data/test_full_unseen.csv'
else:
    test_csv = '../data/test_full.csv'
#test_csv = '../data/test_{}_unseen.csv'.format(gender)


test_rir = '../data/test_rir.txt'


print('loading microphone data')
#train = load_data_rir(train_csv, train_rir)
#dev = load_data_rir(dev_csv, dev_rir)
test = load_data_rir(test_csv, test_rir)



#print ("=== Number of training data: {}".format(len(train)))
print ("=== Number of test data: {}".format(len(test)))


x_test, y_test_3, y_test_12, y_test_rir = list(zip(*test))
x_test = np.array(x_test)

y_test_rir = np.array(y_test_rir)


if args.nclass == 0:
    #y_train = y_train_3
    #y_dev = y_dev_3
    y_test = y_test_3
    classes = classes_3
    experiments = '{}/3class/'.format(gender)
else:
    #y_train = y_train_12
    #y_dev = y_dev_12
    y_test = y_test_12
    classes = classes_12
    experiments = '{}/12class/'.format(gender)

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)


y_test = [cls2label[y] for y in y_test]
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps


# -

model_1 = model_fcnn_rir(12, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
model_2 = model_fcnn_mic(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)

model = model_mic_rir(model_1, model_2)
losses = {
	"model": "mse",
	"model_1": "categorical_crossentropy",
}
metrics = {
    "model": tensorflow.keras.metrics.MeanSquaredError(),
	"model_1": "accuracy",
}
lossWeights = {"model": 1.0, "model_1": 1.0}

weights_path = 'weight/weight_12mic_rir_limit{}_seed{}/full/12class/best_train.hdf5'.format(args.limit, args.seed)
model.load_weights(weights_path)
# +
model.compile(loss=losses, loss_weights=lossWeights,#
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=metrics)
model.summary()
# -





# +

score = model.evaluate(x_test, [y_test,y_test_rir], verbose=0)
#print(score)
print('--- Mic/rir loss:', score[1])
print('- Mic/rir accuracy:', score[3])

file1 = open("record/record_rir_test_{}_review.txt".format(args.limit), "a")  

  
# writing newline character
file1.write("\n")
file1.write(str(score[3]))
file1.close()
