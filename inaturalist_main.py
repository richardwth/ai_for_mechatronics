"""
This code builds a Inception-v3 model using Keras, trains and validates the model for a few epochs
and saves the best model according to validation accuracy.

If you want to run this code, please change all paths in the main.py and misc_fun.py according to your folder structure.

"""
import sys
sys.path.insert(0, 'C:/Users/richa/PycharmProjects/ai_mechatronics/')
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = 'C:/Users/richa/PycharmProjects/ai_mechatronics/Datasets/tfrecords_299/'
FLAGS.DEFAULT_OUT = 'C:/Users/richa/PycharmProjects/ai_mechatronics/Results/iNaturalist2019/'
FLAGS.IMAGE_FORMAT = 'channels_last'
FLAGS.IMAGE_FORMAT_ALIAS = 'NHWC'
import os
if FLAGS.MIXED_PRECISION:  # This line currently has no effects and it's safe to delete
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from GeneralTools.inaturalist_func import read_inaturalist
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.backend import set_session

allow_growth = False
target_size = 299
num_classes = 1010

# read the dataset
dataset_tr, steps_per_tr_epoch = read_inaturalist(
    'train', batch_size=128, target_size=target_size, do_augment=True)
dataset_va, steps_per_va_epoch = read_inaturalist(
    'val', batch_size=256, target_size=target_size, do_augment=True)

if allow_growth:  # allow gpu memory to grow, for debugging purpose, safe to delete
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    set_session(sess)

# load the model
base_model = applications.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(target_size, target_size, 3))
# Freeze some layers  # no need to freeze layers, we fine-tune all layers
# for layer in base_model.layers[:-20]:
#     layer.trainable = False
# Adding custom layers
mdl = Sequential([
    base_model, GlobalAveragePooling2D('channels_last'), Dropout(0.5),
    Dense(num_classes, activation='linear')])
mdl.compile(
    tf.keras.optimizers.RMSprop(lr=0.0001, decay=3e-5),
    loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])

mdl.summary()

# # check point
# checkpoint = ModelCheckpoint(
#     FLAGS.DEFAULT_OUT+'trial_{}_GAPDDropD_RMSprop.h5'.format(target_size), monitor='val_acc', verbose=1,
#     save_best_only=True, save_weights_only=False, mode='auto', period=1)
#
# # do the training
# start_time = time.time()
# history = mdl.fit(
#     dataset_tr.dataset, epochs=2, callbacks=[checkpoint], validation_data=dataset_va.dataset,
#     steps_per_epoch=steps_per_tr_epoch, validation_steps=10, verbose=2)
# duration = time.time() - start_time
# print('\n The training process took {:.1f} seconds'.format(duration))
