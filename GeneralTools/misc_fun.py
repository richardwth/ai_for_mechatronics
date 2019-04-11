"""
This file contains definition for FLAGS.
"""
import tensorflow as tf
flags = tf.flags
FLAGS = tf.flags.FLAGS

# This is a bug of Jupyter notebook
flags.DEFINE_string('f', '', "Empty flag to suppress UnrecognizedFlagError: Unknown command line flag 'f'.")

# local machine configuration
flags.DEFINE_integer('NUM_GPUS', 0, 'Number of GPUs in the local system.')
flags.DEFINE_float('EPSI', 1e-10, 'The smallest positive number to consider.')

# library info
flags.DEFINE_string('TENSORFLOW_VERSION', '1.13.0', 'Version of TensorFlow for the current project.')
flags.DEFINE_string('CUDA_VERSION', '9.0', 'Version of CUDA for the current project.')
flags.DEFINE_string('CUDNN_VERSION', '7.0', 'Version of CuDNN for the current project.')

# working directory info
flags.DEFINE_string('DEFAULT_IN', 'C:/Users/richa/PycharmProjects/ai_mechatronics/Datasets/', 'Default input folder.')
flags.DEFINE_string('DEFAULT_OUT', 'C:/Users/richa/PycharmProjects/ai_mechatronics/Results/', 'Default output folder.')
flags.DEFINE_string(
    'DEFAULT_DOWNLOAD', 'C:/Users/richa/PycharmProjects/ai_mechatronics/Datasets/',
    'Default folder for downloading large datasets.')
flags.DEFINE_string('INCEPTION_V1', None, 'Folder that stores Inception v1 model.')

# data format
flags.DEFINE_string('IMAGE_FORMAT', 'channels_first', 'The format of images by default.')
flags.DEFINE_string('IMAGE_FORMAT_ALIAS', 'NCHW', 'The format of images by default.')

# model hyper-parameters
# flags.DEFINE_string('OPTIMIZER', 'adam', 'The default gradient descent optimizer.')
flags.DEFINE_string('WEIGHT_INITIALIZER', 'default', 'The default weight initialization scheme.')
flags.DEFINE_bool('VERBOSE', True, 'Define whether to print more info during training and test.')
