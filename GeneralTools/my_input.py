"""
This file contains functions and classes to read data
"""
import tensorflow as tf
import numpy as np
import os.path
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


########################################################################
class ReadTFRecords(object):
    def __init__(
            self, filenames, num_features=None, num_labels=0, x_dtype=tf.string, y_dtype=tf.int64, batch_size=64,
            skip_count=0, file_repeat=1, num_epoch=None, file_folder=None,
            num_threads=8, buffer_size=10000, shuffle_file=False):
        """ This function creates a dataset object that reads data from files.

        :param filenames: e.g., cifar
        :param num_features: e.g., 3*64*64
        :param num_labels:
        :param x_dtype: default tf.string, the dtype of features stored in tfrecord file
        :param y_dtype:
        :param num_epoch:
        :param buffer_size:
        :param batch_size:
        :param skip_count: if num_instance % batch_size != 0, we could skip some instances
        :param file_repeat: if num_instance % batch_size != 0, we could repeat the files for k times
        :param num_epoch:
        :param file_folder: if not specified, DEFAULT_IN_FILE_DIR is used.
        :param num_threads:
        :param buffer_size:
        :param shuffle_file: bool, whether to shuffle the filename list

        """
        if file_folder is None:
            file_folder = FLAGS.DEFAULT_IN
        # check inputs
        if isinstance(filenames, str):  # if string, add file location and .tfrecords
            filenames = [os.path.join(file_folder, filenames + '.tfrecords')]
        else:  # if list, add file location and .tfrecords to each element in list
            filenames = [os.path.join(file_folder, file + '.tfrecords') for file in filenames]
        for file in filenames:
            assert os.path.isfile(file), 'File {} does not exist.'.format(file)
        if file_repeat > 1:
            filenames = filenames * int(file_repeat)
        if shuffle_file:
            # shuffle operates on the original list and returns None / does not return anything
            from random import shuffle
            shuffle(filenames)

        # training information
        self.num_features = num_features
        self.num_labels = num_labels
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.batch_size = batch_size
        self.batch_shape = [self.batch_size, self.num_features]
        self.num_epoch = num_epoch
        self.skip_count = skip_count
        # read data,
        dataset = tf.data.TFRecordDataset(filenames)  # setting num_parallel_reads=num_threads decreased the performance
        self.dataset = dataset.map(self.__parser__, num_parallel_calls=num_threads)
        self.iterator = None
        self.buffer_size = buffer_size
        self.scheduled = False
        self.num_threads = num_threads

    ###################################################################
    def __parser__(self, example_proto):
        """ This function parses a single datum

        :param example_proto:
        :return:
        """
        # configure feature and label length
        # It is crucial that for tf.string, the length is not specified, as the data is stored as a single string!
        x_config = tf.FixedLenFeature([], tf.string) \
            if self.x_dtype == tf.string else tf.FixedLenFeature([self.num_features], self.x_dtype)
        if self.num_labels == 0:
            proto_config = {'x': x_config}
        else:
            y_config = tf.FixedLenFeature([], tf.string) \
                if self.y_dtype == tf.string else tf.FixedLenFeature([self.num_labels], self.y_dtype)
            proto_config = {'x': x_config, 'y': y_config}

        # decode examples
        datum = tf.parse_single_example(example_proto, features=proto_config)
        if self.x_dtype == tf.string:  # if input is string / bytes, decode it to float32
            # first decode data to uint8, as data is stored in this way
            datum['x'] = tf.decode_raw(datum['x'], tf.uint8)
            # then cast data to tf.float32
            datum['x'] = tf.cast(datum['x'], tf.float32)
            # cannot use string_to_number as there is only one string for a whole sample
            # datum['x'] = tf.strings.to_number(datum['x'], tf.float32)  # this results in possibly a large number

        # return data
        if 'y' in datum:
            # y can be present in many ways:
            # 1. a single integer, which requires y to be int32 or int64 (e.g, used in tf.gather in cbn)
            # 2. num-class bool/integer/float variables. This form is more flexible as it allows multiple classes and
            # prior probabilities as targets
            # 3. float variables in regression problem.
            # but...
            # y is stored as int (for case 1), string (for other int cases), or float (for float cases)
            # in the case of tf.string and tf.int64, convert to to int32
            if self.y_dtype == tf.string:
                # avoid using string labels like 'cat', 'dog', use integers instead
                datum['y'] = tf.decode_raw(datum['y'], tf.uint8)
                datum['y'] = tf.cast(datum['y'], tf.int32)
            if self.y_dtype == tf.int64:
                datum['y'] = tf.cast(datum['y'], tf.int32)
            return datum['x'], datum['y']
        else:
            return datum['x']

    ###################################################################
    def shape2image(self, channels, height, width, resize=None):
        """ This function shapes the input instance to image tensor

        :param channels:
        :param height:
        :param width:
        :param resize: list of tuple
        :type resize: list, tuple
        :return:
        """

        def image_preprocessor(image):
            # scale image to [-1,1]
            image = tf.subtract(tf.divide(image, 127.5), 1)
            # reshape - note this is determined by how the data is stored in tfrecords, modify with caution
            image = tf.reshape(image, (channels, height, width)) \
                if FLAGS.IMAGE_FORMAT == 'channels_first' else tf.reshape(image, (height, width, channels))
            # resize
            if isinstance(resize, (list, tuple)):
                if FLAGS.IMAGE_FORMAT == 'channels_first':
                    image = tf.transpose(
                        tf.image.resize_images(  # resize only support HWC
                            tf.transpose(image, perm=(1, 2, 0)), resize, align_corners=True), perm=(2, 0, 1))
                else:
                    image = tf.image.resize_images(image, resize, align_corners=True)

            return image

        # do image pre-processing
        if self.num_labels == 0:
            self.dataset = self.dataset.map(
                lambda image_data: image_preprocessor(image_data),
                num_parallel_calls=self.num_threads)
        else:
            self.dataset = self.dataset.map(
                lambda image_data, label: (image_preprocessor(image_data), label),
                num_parallel_calls=self.num_threads)

        # write batch shape
        if isinstance(resize, (list, tuple)):
            height, width = resize
        self.batch_shape = [self.batch_size, height, width, channels] \
            if FLAGS.IMAGE_FORMAT == 'channels_last' else [self.batch_size, channels, height, width]

    ###################################################################
    def scheduler(
            self, batch_size=None, num_epoch=None, shuffle_data=True, buffer_size=None, skip_count=None,
            sample_same_class=False, sample_class=None):
        """ This function schedules the batching process

        :param batch_size:
        :param num_epoch:
        :param buffer_size:
        :param skip_count:
        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class are sampled.
        :param shuffle_data:
        :return:
        """
        if not self.scheduled:
            # update batch information
            if batch_size is not None:
                self.batch_size = batch_size
                self.batch_shape[0] = self.batch_size
            if num_epoch is not None:
                self.num_epoch = num_epoch
            if buffer_size is not None:
                self.buffer_size = buffer_size
            if skip_count is not None:
                self.skip_count = skip_count
            # skip instances
            if self.skip_count > 0:
                print('Number of {} instances skipped.'.format(self.skip_count))
                self.dataset = self.dataset.skip(self.skip_count)
            # shuffle
            if shuffle_data:
                self.dataset = self.dataset.shuffle(self.buffer_size)
            # set batching process
            if sample_same_class:
                if sample_class is None:
                    print('Caution: samples from the same class at each call.')
                    group_fun = tf.contrib.data.group_by_window(
                        key_func=lambda data_x, data_y: data_y,
                        reduce_func=lambda key, d: d.batch(self.batch_size),
                        window_size=self.batch_size)
                    self.dataset = self.dataset.apply(group_fun)
                else:
                    print('Caution: samples from class {}. This should not be used in training'.format(sample_class))
                    self.dataset = self.dataset.filter(lambda x, y: tf.equal(y[0], sample_class))
                    self.dataset = self.dataset.batch(self.batch_size)
            else:
                self.dataset = self.dataset.batch(self.batch_size)
            # self.dataset = self.dataset.padded_batch(batch_size)
            if self.num_epoch is None:
                self.dataset = self.dataset.repeat()
            else:
                FLAGS.print('Num_epoch set: {} epochs.'.format(num_epoch))
                self.dataset = self.dataset.repeat(self.num_epoch)

            self.iterator = self.dataset.make_one_shot_iterator()
            self.scheduled = True

    ###################################################################
    def next_batch(self, sample_same_class=False, sample_class=None, shuffle_data=True):
        """ This function generates next batch

        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class are sampled.
            Note that class_label should not be provided during training.
        :param shuffle_data:
        :return:
        """
        if self.num_labels == 0:
            if not self.scheduled:
                self.scheduler(shuffle_data=shuffle_data)
            x_batch = self.iterator.get_next()

            x_batch.set_shape(self.batch_shape)

            return {'x': x_batch}
        else:
            if sample_class is not None:
                assert isinstance(sample_class, (np.integer, int)), \
                    'class_label must be integer.'
                assert sample_class < self.num_labels, \
                    'class_label {} is larger than maximum value {} allowed.'.format(
                        sample_class, self.num_labels - 1)
                sample_same_class = True
            if not self.scheduled:
                self.scheduler(
                    shuffle_data=shuffle_data, sample_same_class=sample_same_class,
                    sample_class=sample_class)
            x_batch, y_batch = self.iterator.get_next()

            x_batch.set_shape(self.batch_shape)
            y_batch.set_shape([self.batch_size, self.num_labels])

            return {'x': x_batch, 'y': y_batch}
