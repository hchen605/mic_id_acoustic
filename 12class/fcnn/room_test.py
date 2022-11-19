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
parser.add_argument("--unseen", type=int, default=0, help="seen (0), unseen(1)")
parser.add_argument("--nclass", type=int, default=0, help="mic (0), room(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")

args = parser.parse_args()

#tensorflow.reset_default_graph()
#os.environ['PYTHONHASHSEED']='0'#str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# +
classes_3 = ['sma','med','lar']
classes_12 = ['sma1','sma1','med1','med2','med3','lar1','lar2','lar2']





if args.unseen:
    test_csv = '../data/test_room_unseen_d3.csv'
else:
    test_csv = '../data/test_room_seen.csv'




print('loading microphone data')
#train = load_data(train_csv)
#dev = load_data(dev_csv)
test = load_data(test_csv)



#print ("=== Number of training data: {}".format(len(train)))
print ("=== Number of test data: {}".format(len(test)))

x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)



if args.nclass == 0:
    y_test = y_test_3
    classes = classes_3
else:
    y_test = y_test_12
    classes = classes_test


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

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model = model_xvector(num_classes)
weights_path = 'weight/weight_room_limit{}/3class/best.hdf5'.format(args.limit)
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

#model.summary()


# +

score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])


file1 = open("./record/record_room_{}.txt".format(args.limit), "a") 
  
# writing newline character
file1.write("\n")
file1.write(str(score[1]))
file1.close()


#confusion matrix

y_pred = model.predict(x_test)
y_test_ = np.argmax(y_test, axis=-1)
y_pred_ = np.argmax(y_pred, axis=-1)

data_df = pd.read_csv(test_csv, sep='\t')   
wavpath = data_df['filename'].tolist()

indices = [i for i,v in enumerate(y_pred) if y_pred_[i]!=y_test_[i]]
#subset_of_wrongly_predicted = [x_test for i in indices ]
#subset_of_wrongly_predicted = [test_csv[i] for i in indices ]
file1 = open("./record/room_mems_unseen_error.txt".format(args.limit), "a") 

for i in indices:
    file1.write(wavpath[i])
    file1.write("\n")
file1.close()
#print(wavpath[i] for i in indices)

cm = confusion_matrix(y_test_, y_pred_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('AttCNN Room Size Prediction')
plt.show()
#plt.savefig('./music_log/cm_{}_{}.pdf'.format(mic,target))
#plt.savefig('./confusion/cm_room_{}_seed{}.pdf'.format(args.limit, args.seed))
plt.savefig('./confusion/cm_room_{}_unseen_D3.pdf'.format(args.limit, args.seed))
