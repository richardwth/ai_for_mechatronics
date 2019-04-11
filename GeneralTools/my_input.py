"""
This file contains functions and classes to read data
"""
import tensorflow as tf
import numpy as np
from GeneralTools.misc_fun import FLAGS


########################################################################
def create_placeholder(tensor, dtype=None):
    """ This function creates a placeholder for tensor.
    When the input tensor is a list/tuple of tensors, it creates a list of placeholders

    :param tensor: input tensor must be one of the following types: np.ndarray, tuple or list of np.ndarray
    :param dtype: a basic TensorFlow DType defined in tf.dtypes.DType.
    If provided, the placeholder will have the given dtype;
    otherwise, the dtype is inferred.
    When tensor is a list or tuple, dtype must be a basic DType, list or tuple.
    :return:
    """
    if isinstance(tensor, np.ndarray):
        if dtype is None:
            if tensor.dtype in {np.float32, np.float64, np.float16, np.float}:
                placeholder = tf.placeholder(tf.float32, tensor.shape)
            elif tensor.dtype in {np.int, np.int32, np.int64}:
                placeholder = tf.placeholder(tf.int32, tensor.shape)
            else:
                raise NotImplementedError('The dtype {} is not implemented.'.format(tensor.dtype))
        else:
            placeholder = tf.placeholder(dtype, tensor.shape)
    elif isinstance(tensor, tf.Tensor):
        raise TypeError('The input to placeholder cannot be tf.Tensor.')
    elif isinstance(tensor, (list, tuple)):
        if isinstance(dtype, (list, tuple)):
            placeholder = tuple([create_placeholder(
                single_tensor, single_dtype) for single_tensor, single_dtype in zip(tensor, dtype)])
        else:
            placeholder = tuple([create_placeholder(
                single_tensor, dtype) for single_tensor in tensor])
    else:
        raise NotImplementedError(
            'Placeholder can only be created for numpy array, tf.Tensor, list or tuple')

    return placeholder


########################################################################
class DatasetFromTensor(object):
    """
    This class provides an example of defining a customized input pipeline.
    """
    def __init__(self, data, dtype=None, map_func=None, name='Input'):
        """ Initialize a dataset and applies map_func

        :param data: Input data must be from the following types: numpy array,
        tuple and list. Placeholder will be automatically created for data or each element in data.
        :param dtype: the dtype of tensor, can be a basic basic DType, list or tuple.
        :param map_func: common map_func includes scale and reshape
        :param name:
        """
        self.name = name
        # check input types
        assert isinstance(data, (np.ndarray, tf.Tensor, tuple, list)), \
            '{}: Data must be from the following types: numpy array, tuple and list.'.format(self.name)

        with tf.name_scope(name=self.name) as self.scope:
            # create placeholder and initialize the dataset
            self.placeholder = create_placeholder(data, dtype)
            self.feed_dict = {}
            for key, value in zip(self.placeholder, data):
                self.feed_dict[key] = value
            self.num_samples = data.shape[0] if isinstance(data, np.ndarray) else data[0].shape[0]
            self.dataset = tf.data.Dataset.from_tensor_slices(self.placeholder)
            # apply map_func
            if map_func is not None:
                self.dataset = self.dataset.map(map_func)
        self.iterator = None

        if FLAGS.VERBOSE:
            print('{} dataset output type is {}'.format(
                self.name, self.dataset.output_types))
            print('{} dataset output shape is {}'.format(
                self.name, self.dataset.output_shapes))

    def schedule(self, batch_size, num_epoch=-1, skip_count=None, shuffle=True, buffer_size=10000):
        with tf.name_scope(self.scope):
            if skip_count is None:
                skip_count = self.num_samples % batch_size
            if skip_count > 0:
                self.dataset = self.dataset.skip(skip_count)
                if FLAGS.VERBOSE:
                    print('{}: Number of {} instances skipped.'.format(
                        self.name, skip_count))
            # shuffle
            if shuffle:
                self.dataset = self.dataset.shuffle(buffer_size)
            # make batch
            self.dataset = self.dataset.batch(batch_size)
            # repeat datasets for num_epoch
            self.dataset = self.dataset.repeat(num_epoch)
            # initialize an iterator
            self.iterator = self.dataset.make_initializable_iterator()

        return self  # facilitate method cascading

    def next(self, batch_size=None):
        if self.iterator is None:
            assert batch_size is not None, \
                '{}: Batch size must be provided.'.format(self.name)
            self.schedule(batch_size)
        # somehow set_shape does not work as expected.
        # data_batch = self.iterator.get_next()
        # output_shapes = [[batch_size] + output_shape.as_list()[1:] for output_shape in self.dataset.output_shapes]
        # for index, each_shape in enumerate(output_shapes):
        #     data_batch[index].set_shape(each_shape)
        # print(data_batch)
        # return data_batch
        with tf.name_scope(self.scope):
            return self.iterator.get_next()

    def initializer(self):
        assert self.iterator is not None, \
            '{}: Batch must be provided.'.format(self.name)
        return self.iterator.initializer, self.feed_dict
