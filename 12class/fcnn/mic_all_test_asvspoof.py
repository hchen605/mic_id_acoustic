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
#classes_3 = ['C','D','M','P']
classes_3 = ['P','X']
classes_12 = ['P1','P2','X']
genders = ['full', 'female', 'male']
classes_room = ['large','medium','small']



gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12


test_csv = '../data/ood/test_asvspoof_few.csv'
#test_csv_id = '../data/ood/test_asvspoof_p_id.csv'
test_csv_id = '../data/ood/test_full_mobile_clo_4th_abstention_p1_dev_.csv'
dev_csv = '../data/ood/dev_asvspoof_abstention_p.csv'

#test_csv = '../data/test_full_mobile_clo_4th_sp1_test.csv'



print('loading microphone data')
test = load_data(test_csv)
test_id = load_data(test_csv_id)
dev = load_data(dev_csv)

print ("=== Number of test data: {}".format(len(test)))


x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)

x_test_id, y_test_3_2, y_test_12_2 = list(zip(*test_id))
x_test_id = np.array(x_test_id)

x_dev, y_test_3_3, y_test_12_3 = list(zip(*dev))
x_dev = np.array(x_dev)


if args.nclass == 0:
    y_test = y_test_3
    classes = classes_3
    experiments = '{}/3class/'.format(gender)
else:
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
input_length = 48000

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)


weights_path = 'weight/weight_asvspoof_abstention_p1p2_id/' + "best.hdf5"
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


model.summary()

'''
score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])

file1 = open("./record/record_full_mobile_clo_class_{}_limit_{}_asvspoof.txt".format(args.nclass, args.limit), "a") 
  
# writing newline character
file1.write("\n")
file1.write(str(score[1]))
file1.close()
'''

#confusion matrix
#np.save('../data/ood/test_p_abstention.npy',y_test)
y_pred = model.predict(x_test)
y_pred_c = np.squeeze(y_pred)
#print(y_pred_c)
#print(y_pred_c.shape)
np.save('../data/ood/test_asvspoof_in_p2_abstention_few_label.npy',y_pred_c[:,1])
#print(y_pred_c[:,1])
#print(np.mean(y_pred_c[:,1]))

y_ood = y_pred_c[:,2]

y_test_ = np.argmax(y_test, axis=-1)
y_pred_ = np.argmax(y_pred, axis=-1)

y_pred = model.predict(x_test_id)
y_pred_c = np.squeeze(y_pred)
#print(y_pred_c)
#print(y_pred_c.shape)
#np.save('../data/ood/test_asvspoof_p1_in_p1p2_abstention_id.npy',y_pred_c[:,2])
#print(y_pred_c[:,2])
#print(np.mean(y_pred_c[:,2]))

y_id = y_pred_c[:,2]


'''
y_pred = model.predict(x_dev)
y_pred_c = np.squeeze(y_pred)
#print(y_pred_c)
print(y_pred_c.shape)
np.save('../data/ood/dev_asvspoof_p_abstention_ood.npy',y_pred_c[:,0])
'''

scores = np.array(
    np.concatenate([
     #np.max(y_id,axis=-1),
     #np.max(y_ood,axis=-1),
     y_id,
     y_ood,
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