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
from models.small_fcnn_att import model_fcnn, model_fcnn_Prelogits
from models.xvector import model_xvector
from models.attRNN import AttRNN, AttRNN_pre

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import entropy


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
#classes_12 = ['P1','P2','P3','P4','X']
classes_12 = ['P1','P2','X']
genders = ['full', 'female', 'male']
classes_room = ['large','medium','small']



gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12

train_csv = '../data/ood/train_full_mobile_clo_4th_abstention_p_trial.csv'
test_csv = '../data/ood/test_full_mobile_clo_4th_abstention_p1p2_dev_.csv'
test_csv_2 = '../data/ood/test_full_mobile_clo_4th_abstention_p_dev.csv'
dev_csv = '../data/ood/dev_full_mobile_clo_4th_abstention_p_trial_p1p2_dev_id.csv'
dev_csv_2 = '../data/ood/dev_full_mobile_clo_4th_abstention_p_trial_p1p2_dev_ood.csv'

#test_csv = '../data/test_full_mobile_clo_4th_sp1_test.csv'



print('loading microphone data')
test = load_data(test_csv)
test_2 = load_data(test_csv_2)
dev = load_data(dev_csv)
dev_2 = load_data(dev_csv_2)
#train = load_data(train_csv)

print ("=== Number of test data: {}".format(len(test)))


x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)
x_test_2, y_test_3_2, y_test_12_2 = list(zip(*test_2))
x_test_2 = np.array(x_test_2)
x_dev, y_dev_3, y_dev_12 = list(zip(*dev))
x_dev = np.array(x_dev)
x_dev_2, y_dev_3_2, y_dev_12_2 = list(zip(*dev_2))
x_dev_2 = np.array(x_dev_2)
#x_train, y_train_3, y_train_12 = list(zip(*train))
#x_train = np.array(x_train)



if args.nclass == 0:
    y_test = y_test_3
    classes = classes_3
    #y_train = y_train_3
    experiments = '{}/3class/'.format(gender)
else:
    y_test = y_test_12
    #y_train = y_train_12
    classes = classes_12
    experiments = '{}/12class/'.format(gender)

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_test = [cls2label[y] for y in y_test]
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
#y_train = [cls2label[y] for y in y_train]
#y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)



if args.nclass == 0:
    y_test_2 = y_test_3_2
    classes = classes_3
    experiments = '{}/3class/'.format(gender)
else:
    y_test_2 = y_test_12_2
    classes = classes_12
    experiments = '{}/12class/'.format(gender)


#y_test_2 = [cls2label[y] for y in y_test_2]
#y_test_2 = keras.utils.to_categorical(y_test_2, num_classes=num_classes)

# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps
input_length = 48000

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model_prelogic = model_fcnn_Prelogits(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)

weights_path = 'weight/weight_full_mobile_limit{}_seed{}_abstention_p_id_ood_p1p2_dev/'.format(args.limit, args.seed)+ experiments + "best.hdf5"
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
'''
model_prelogic.load_weights(weights_path)

model_prelogic.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
'''

model.summary()


score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])

'''
file1 = open("./record/record_full_mobile_clo_class_{}_limit_{}_abstention_p1_dev.txt".format(args.nclass, args.limit), "a") 
  
# writing newline character
file1.write("\n")
file1.write(str(score[1]))
file1.close()
'''

#confusion matrix
#np.save('../data/ood/test_p_abstention.npy',y_test)
#y_prelogic_id = model_prelogic.predict(x_test)
#y_prelogit_ood = model_prelogic.predict(x_test_2)
#y_prelogic_id_train = model_prelogic.predict(x_train)

y_id = model.predict(x_test)
y_ood = model.predict(x_test_2)
y_dev_id = model.predict(x_dev)
y_dev_ood = model.predict(x_dev_2)

y_id_c = np.squeeze(y_id)
np.save('../data/ood/test_p_abstention_trial_10_p1p2_dev_id.npy',y_id_c[:,2])
y_ood_c = np.squeeze(y_ood)
np.save('../data/ood/test_p_abstention_trial_10_p1p2_dev_ood.npy',y_ood_c[:,2])
y_dev_id = np.squeeze(y_dev_id)
#np.save('../data/ood/dev_p_abstention_trial_10_p3p4_dev_id.npy',y_dev_id[:,2])
y_dev_ood = np.squeeze(y_dev_ood)
#np.save('../data/ood/dev_p_abstention_trial_10_p3p4_dev_ood.npy',y_dev_ood[:,2])
#print(y_id)
#print(y_ood)
#print(y_prelogic_id_train)
#print(y_id[:,4])
#print(y_ood[:,4])

scores = np.array(
    np.concatenate([
     #np.max(y_id,axis=-1),
     #np.max(y_ood,axis=-1),
     y_id[:,2],
     y_ood[:,2],
    ],axis=0)
)

onehots = np.array(
    [0]*len(y_id)+[1]*len(y_ood)
)

auroc, to_replot_dict = get_auroc(
    onehots, 
    scores, 
    make_plot=True,
    add_to_title="ViT-L_16 on CIFAR-100 vs CIFAR-10\nMax of Softmax Probs",
    swap_classes=True,
    )

print('auroc: ', auroc)
'''
onehots, scores, description, maha_intermediate_dict = get_scores(
        np.array(y_prelogic_id_train)[:,:],
        y_train,
        np.array(y_prelogic_id)[:,:],
        np.array(y_prelogit_ood)[:,:],
        indist_classes=4,
        subtract_mean = False,
        normalize_to_unity = False,
        subtract_train_distance = False,
    )

'''

