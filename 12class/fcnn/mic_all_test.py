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
from models.small_fcnn_att import model_fcnn, model_fcnn_pre
#
#from models.attRNN import AttRNN, AttRNN_pre

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
parser.add_argument("--nclass", type=int, default=0, help="3class (0), 12class(1)")
parser.add_argument("--limit", type=int, default=400, help="number of data")
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
classes_room = ['large','medium','small']



gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12


test_csv = '../data/test_mic_unseen.csv'
#test_csv = '../data/test_full_mobile_clo_apple_samsung_2.csv'
#test_csv = '../data/test_mic_small_1m.csv'



print('loading microphone data')
test = load_data(test_csv)

print ("=== Number of test data: {}".format(len(test)))


x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)



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
#model = model_xvector(num_classes)
#model = AttRNN(num_classes, input_length)
#model = AttRNN_pre(num_classes, input_length)

#weights_path = 'weight/weight_full_mobile_limit{}_seed{}_18class_iphone_rir_to_others_clo/'.format(args.limit, args.seed)+ experiments + "best.hdf5"
weights_path = 'weight/weight_18class_test_dist_student/'+ experiments + "best.hdf5"
#weights_path = 'weight/weight_full_mobile_limit400_seed0_mic_18class_5m_7m_9m_rir_0p66_clo/'+ experiments + "best.hdf5"
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


model.summary()


score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])


#file1 = open("./record/record_mic_{}_limit_{}_mic_rir.txt".format(args.nclass, args.limit), "a") 
  
# writing newline character
#file1.write("\n")
#file1.write(str(score[1]))
#file1.close()



#confusion matrix
y_pred = model.predict(x_test)
y_test_ = np.argmax(y_test, axis=-1)
y_pred_ = np.argmax(y_pred, axis=-1)
#print(y_test_)
#print(y_pred_)
'''
cm = confusion_matrix(y_test_, y_pred_)
#print(cm)

classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M2','M3','P1','P2','P3','P4','P5','P6']
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_12)
disp.plot(cmap=plt.cm.Blues)
plt.title('Microphone Classification')
plt.show()
#plt.savefig('./music_log/cm_{}_{}.pdf'.format(mic,target))
plt.savefig('./confusion/cm_full_mobile_class_{}_limit_{}_seed_{}_samsung_iphone_rir_pre_2000_clo_10.pdf'.format(args.nclass, args.limit, args.seed))
'''