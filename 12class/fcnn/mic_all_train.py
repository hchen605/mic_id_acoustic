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
#from models.xvector import model_xvector
#from models.attRNN import AttRNN, AttRNN_pre
#from models.fcnn_att import model_fcnn_dcase

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

#classes_3 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511']
#classes_12 = classes_3
# -

gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12


train_csv = '../data/train_full_mobile_clo_4th.csv'
dev_csv = '../data/dev_full_mobile_clo_4th.csv'
#train_csv = '../data/train_audioset.csv'
#dev_csv = '../data/dev_audioset.csv'
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

#mark if audioset
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
#weights_path = '/home/hsinhung/mic_acoustic/12class/fcnn/weight/weight_full_mobile_limit400_seed0_audioset/full/3class/best.hdf5'
#model.load_weights(weights_path)
#model = AttRNN_pre(num_classes, input_length)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])


model.summary()

# Checkpoints
if not os.path.exists('weight/weight_full_mobile_limit{}_seed{}_unseen/'.format(args.limit, args.seed)+experiments):
    os.makedirs('weight/weight_full_mobile_limit{}_seed{}_unseen/'.format(args.limit, args.seed)+experiments)

#save_path = "weight/weight_full_mobile_limit{}_seed{}/".format(args.limit, args.seed)+ experiments + "{epoch:02d}-{val_accuracy:.4f}.hdf5"
save_path = "weight/weight_full_mobile_limit{}_seed{}_unseen/".format(args.limit, args.seed)+ experiments + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks = [checkpoint]


# Training
exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. Acc: ", max(exp_history.history['val_accuracy']), " At Epoch of ", np.argmax(exp_history.history['val_accuracy'])+1)

# +

