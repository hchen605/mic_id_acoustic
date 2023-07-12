import os
import sys
sys.path.append("..")
import argparse

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam

from utils import *
from funcs import *

from ts_dataloader import *
from models.small_fcnn_att import model_fcnn, model_fcnn_specaug
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
tf.random.set_seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)
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
def lm_loss(y_true, y_pred):
    epsilon = 1e-5
    entropy = -y_pred * tf.math.log(y_pred + epsilon)
    entropy = tf.math.reduce_sum(entropy, axis=1)

    msoftmax = tf.math.reduce_mean(y_pred, axis=0)
    gentropy_loss = tf.math.reduce_sum(-msoftmax * tf.math.log(msoftmax + epsilon))
    entropy -= gentropy_loss

    cce = keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    loss += entropy

    return loss

gender = genders[args.gender]


if args.nclass == 0:
    classes_test = classes_3
else:
    classes_test = classes_12


train_csv = '../data/test_mic_dist_full.csv'
dev_csv = '../data/test_mic_dist_full.csv'
#dev_csv = '../data/dev_mic_18class_5m_7m_9m_rir.csv'
train_csv_labeled = '../data/train_mic_18class_5m_7m_9m_rir_0p66_clo.csv'
dev_csv_labeled = '../data/dev_mic_18class_5m_7m_9m_rir_0p66_clo.csv'
#test_csv = '../data/test_full_mobile_clo.csv'



print('loading microphone data')
train = load_data(train_csv)
dev = load_data(dev_csv)
train_labeled = load_data(train_csv_labeled)
dev_labeled = load_data(dev_csv_labeled)
#test = load_data(test_csv)


print ("=== Number of training data: {}".format(len(train)))
print ("=== Number of training labeled data: {}".format(len(train_labeled)))
print('Data processing ..')

x_train, y_train_3, y_train_12 = list(zip(*train))
x_dev, y_dev_3, y_dev_12 = list(zip(*dev))
x_train = np.array(x_train)
x_dev = np.array(x_dev)

x_train_labeled, y_train_labeled_3, y_train_labeled_12 = list(zip(*train_labeled))
x_dev_labeled, y_dev_labeled_3, y_dev_labeled_12 = list(zip(*dev_labeled))
x_train_labeled = np.array(x_train_labeled)
x_dev_labeled = np.array(x_dev_labeled)


y_train = y_train_12
y_dev = y_dev_12
classes = classes_12
experiments = '{}/12class/'.format(gender)

y_train_labeled = y_train_labeled_12
y_dev_labeled = y_dev_labeled_12

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]
y_dev = [cls2label[y] for y in y_dev]
y_train_labeled = [cls2label[y] for y in y_train_labeled]
y_dev_labeled = [cls2label[y] for y in y_dev_labeled]

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)
y_train_labeled = keras.utils.to_categorical(y_train_labeled, num_classes=num_classes)
y_dev_labeled = keras.utils.to_categorical(y_dev_labeled, num_classes=num_classes)

print('Data processing finished..')

# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 64
epochs = args.eps
pseudo_th = 0.1

mask_num = []

# Model
#teacher_model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#student_model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
teacher_model = model_fcnn_specaug(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0, mask_prob=0.1)
student_model = model_fcnn_specaug(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0, mask_prob=0.2)

#model = model_xvector(num_classes)
weights_path = 'weight/weight_full_mobile_limit400_seed0_mic_18class_5m_7m_9m_rir_0p66_clo/'+ experiments + "best.hdf5"
teacher_model.load_weights(weights_path)

#softmax_out = teacher_model.predict(x_dev)
#pseudo_label = np.argmax(softmax_out,axis=-1)
#y_true = np.load('../data/ood/pseudo_update_4.npy')
#y_true = np.load('../data/test_dist_pseudo_label.npy')
#y_pseudo = keras.utils.to_categorical(pseudo_label, num_classes=num_classes)

teacher_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
student_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

teacher_model.summary()

# Create a dataset object
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

train_labeled_dataset = tf.data.Dataset.from_tensor_slices((x_train_labeled, y_train_labeled))
train_labeled_dataset = train_labeled_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
val_dataset = val_dataset.batch(batch_size)

val_labeled_dataset = tf.data.Dataset.from_tensor_slices((x_dev_labeled, y_dev_labeled))
val_labeled_dataset = val_labeled_dataset.batch(batch_size)

# Checkpoints
if not os.path.exists('weight/weight_18class_test_dist_student/'+experiments):
    os.makedirs('weight/weight_18class_test_dist_student/'+experiments)

if not os.path.exists('weight/weight_18class_test_dist_teacher/'+experiments):
    os.makedirs('weight/weight_18class_test_dist_teacher/'+experiments)

#save_path = "weight/weight_full_mobile_limit{}_seed{}/".format(args.limit, args.seed)+ experiments + "{epoch:02d}-{val_accuracy:.4f}.hdf5"
save_path = "weight/weight_18class_test_dist_student/"+ experiments + "best.hdf5"
save_path_t = "weight/weight_18class_test_dist_teacher/"+ experiments + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [checkpoint]


# Training
loss_fn = tf.keras.losses.CategoricalCrossentropy()
ema = tf.train.ExponentialMovingAverage(decay=0.999)
optimizer = tf.keras.optimizers.Adam()
best_val_loss = float('inf')

for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    train_loss = 0
    num_batches = 0
    #mask_sum = 0
    # iterate over batchs
    for batch_idx, (x_labeled, y_labeled) in enumerate(train_labeled_dataset):
        #loss = train_on_batch()
        with tf.GradientTape() as tape:
            y_pred_labeled = student_model(x_labeled, training=True)
            loss_labeled = loss_fn(y_labeled, y_pred_labeled)
        grads = tape.gradient(loss_labeled, student_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, student_model.trainable_weights))
        num_batches += 1
        train_loss += loss_labeled
        if batch_idx % 20 == 0:
            print("-- Batch %d: training labeled loss = %.4f" % (batch_idx, loss_labeled))
    
    for batch_idx, (x, y) in enumerate(train_dataset):
        softmax_out = teacher_model.predict(x)
        #pseudo_label = np.argmax(softmax_out,axis=-1)
        y_t_max = np.max(softmax_out, axis=1)
        # find confident output
        mask = y_t_max >= pseudo_th
        x_mask = x[mask]
        y_mask = softmax_out[mask]
        # binarize
        y_mask[y_mask < pseudo_th] = 0
        y_mask[y_mask > 0] = 1

        mask_sum = mask_sum + np.sum(mask)

        with tf.GradientTape() as tape:
            y_pred_mask = student_model(x_mask, training=True)
            loss_mask = loss_fn(y_mask, y_pred_mask)
        if loss_mask == 0.0:
            break
        grads = tape.gradient(loss_mask, student_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, student_model.trainable_weights))
        train_loss += loss_mask
        if batch_idx % 20 == 0:
            print("-- Batch %d: training unlabeled loss = %.4f" % (batch_idx, loss_mask))
    train_loss /= (num_batches+batch_idx)
    #mask_num.append(mask_sum)
    #print('-- mask_sum: ', mask_sum)

    #validation
    val_loss = 0
    num_batches = 0
    for batch_idx, (x_labeled, y_labeled) in enumerate(val_labeled_dataset):
        #loss = train_on_batch()
        y_pred_labeled = student_model(x_labeled, training=False)
        loss_labeled = loss_fn(y_labeled, y_pred_labeled)
        num_batches += 1
        val_loss += loss_labeled
    
    for batch_idx, (x, y) in enumerate(val_dataset):
        softmax_out = teacher_model.predict(x)
        #pseudo_label = np.argmax(softmax_out,axis=-1)
        y_t_max = np.max(softmax_out, axis=1)
        # find confident output
        mask = y_t_max >= pseudo_th
        x_mask = x[mask]
        y_mask = softmax_out[mask]
        # binarize
        y_mask[y_mask < pseudo_th] = 0
        y_mask[y_mask > 0] = 1

        y_pred_mask = student_model(x_mask, training=True)
        loss_mask = loss_fn(y_mask, y_pred_mask)
    
        val_loss += loss_mask

    val_loss /= (num_batches+batch_idx)
    print("Epoch %d: training loss = %.4f, validation loss = %.4f" % (epoch+1, train_loss, val_loss))
    
    #save
    if val_loss < best_val_loss:
        student_model.save_weights(save_path)
        teacher_model.save_weights(save_path_t)
        best_val_loss = val_loss
        print(f"Epoch {epoch+1}: saved best model with val_loss={val_loss:.4f}")
    # Update the teacher model with exponential moving average
    #ema_decay = 0.999
    #if epoch > 20:
    #    for weight, ema_weight in zip(student_model.weights, teacher_model.weights):
    #        ema_weight.assign(ema_decay * ema_weight + (1 - ema_decay) * weight)

#print(mask_num)     
#exp_history = student_model.fit(x_train, y_pseudo, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
#              validation_data=(x_dev, y_dev), callbacks=callbacks)

#print("=== Best Val. loss: ", max(exp_history.history['val_loss']), " At Epoch of ", np.argmin(exp_history.history['val_loss'])+1)

# +

def lm_loss(y_true, y_pred):
    epsilon = 1e-5
    entropy = -y_pred * tf.math.log(y_pred + epsilon)
    entropy = tf.math.reduce_sum(entropy, axis=1)

    msoftmax = tf.math.reduce_mean(y_pred, axis=0)
    gentropy_loss = tf.math.reduce_sum(-msoftmax * tf.math.log(msoftmax + epsilon))
    entropy -= gentropy_loss

    cce = keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    loss += entropy

    return loss