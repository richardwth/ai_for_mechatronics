"""
This file contains FLAGS definitions.
"""
import tensorflow as tf
flags = tf.flags
FLAGS = tf.flags.FLAGS

# This is a bug of Jupyter notebook
flags.DEFINE_string('f', '', "Empty flag to suppress UnrecognizedFlagError: Unknown command line flag 'f'.")

# local machine configuration
flags.DEFINE_integer('NUM_GPUS', 1, 'Number of GPUs in the local system.')
flags.DEFINE_float('EPSI', 1e-10, 'The smallest positive number to consider.')
flags.DEFINE_bool('MIXED_PRECISION', True, 'Whether to use automatic mixed precision.')

# library info
flags.DEFINE_string('TENSORFLOW_VERSION', '1.13.0', 'Version of TensorFlow for the current project.')
flags.DEFINE_string('CUDA_VERSION', '10.0', 'Version of CUDA for the current project.')
flags.DEFINE_string('CUDNN_VERSION', '7.3', 'Version of CuDNN for the current project.')

# working directory info
flags.DEFINE_string('SYSPATH', '/home/richard/PycharmProjects/myNN/', 'Default working folder.')
flags.DEFINE_string('DEFAULT_IN', '/media/richard/ExtraStorage/Data/', 'Default input folder.')
flags.DEFINE_string('DEFAULT_OUT', '/home/richard/PycharmProjects/myNN/Results/', 'Default output folder.')
flags.DEFINE_string(
    'DEFAULT_DOWNLOAD', '/media/richard/My Book/MyBackup/Data/',
    'Default folder for downloading large datasets.')
flags.DEFINE_string(
    'INCEPTION_V1',
    '/home/richard/PycharmProjects/myNN/Code/inception_v1/inceptionv1_for_inception_score.pb',
    'Folder that stores Inception v1 model.')
flags.DEFINE_string(
    'INCEPTION_V3',
    '/home/richard/PycharmProjects/myNN/Code/inception_v3/classify_image_graph_def.pb',
    'Folder that stores Inception v1 model.')

# plotly account
flags.DEFINE_string('PLT_ACC', 'Richard_wth', 'Matplotlib acc to synchronize the plots.')
flags.DEFINE_string('PLT_KEY', 'cqBAQrgDsHm1blKmVVn8', 'Matplotlib password.')

# data format
flags.DEFINE_string('IMAGE_FORMAT', 'channels_first', 'The format of images by default.')
flags.DEFINE_string('IMAGE_FORMAT_ALIAS', 'NCHW', 'The format of images by default.')

# model hyper-parameters
flags.DEFINE_string(
    'WEIGHT_INITIALIZER', 'default',
    'The default weight initialization scheme. Could also be sn_paper, pg_paper')
flags.DEFINE_string(
    'SPECTRAL_NORM_MODE', 'default',
    'The default power iteration method. Default is to use PICO. Could also be sn_paper.')
flags.DEFINE_bool('VERBOSE', True, 'Define whether to print more info during training and test.')
