"""
This file contains functions and classes for saving, reading and processing the iNaturalist 2019 dataset

"""
from GeneralTools.misc_fun import FLAGS
import sys
import time
import os.path
import tensorflow as tf
import numpy as np


########################################################################
# define macro
# FloatList, Int64List and BytesList are three base feature types
def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
        if isinstance(value, float) else tf.train.FloatList(value=value))


def _int64_feature(value):  # numpy int is not int!
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
        if isinstance(value, int) else tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
        if isinstance(value, (str, bytes)) else tf.train.BytesList(value=value))
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


########################################################################
def images_to_tfrecords(image_names, output_filename, num_images_per_tfrecord, image_class=None, target_size=299):
    """ This function converts images listed in the image_names to tfrecords files

    :param image_names: a list of strings like ['xxx.jpg', 'xxx.jpg', 'xxx.jpg', ...]
    :param output_filename: 'train'
    :param num_images_per_tfrecord: integer
    :param image_class: class label for each image, a list like [1, 34, 228, ...]
    :param target_size: the size of images after padding and resizing
    :return:
    """
    from PIL import Image, ImageOps
    from io import BytesIO

    num_images = len(image_names)
    # iteratively handle each image
    writer = None
    start_time = time.time()
    for image_index in range(num_images):
        # retrieve a single image
        im_loc = os.path.join(FLAGS.DEFAULT_DOWNLOAD, image_names[image_index])
        im_cla = image_class[image_index] if isinstance(image_class, list) else None
        im = Image.open(im_loc)

        # resize the image
        old_size = im.size
        ratio = float(target_size) / max(old_size)
        if not ratio == 1.0:
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.LANCZOS)

        # zero-pad the images
        new_size = im.size
        delta_w = target_size - new_size[0]
        delta_h = target_size - new_size[1]
        if delta_w < 0 or delta_h < 0:
            raise AttributeError('The target size is smaller than the image size {}.'.format(new_size))
        elif delta_w > 0 or delta_h > 0:
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            im = ImageOps.expand(im, padding)

        # if image not RGB format, convert to RGB
        # This is done in case the image is greyscale
        if im.mode != 'RGB':
            im = im.convert('RGB')

        # convert the full image data to the jpeg compressed string to reduce tfrecord size
        with BytesIO() as fp:
            im.save(fp, format="JPEG")
            im_string = fp.getvalue()

        # im.show('a new image')
        # save to tfrecords
        if image_index % num_images_per_tfrecord == 0:
            file_out = "{}_{:03d}.tfrecords".format(output_filename, image_index // num_images_per_tfrecord)
            # print("Writing on:", file_out)
            writer = tf.python_io.TFRecordWriter(file_out)
        if image_class is None:
            # for test set, the labels are unknown and not provided
            instance = tf.train.Example(
                features=tf.train.Features(feature={
                    'x': _bytes_feature(im_string)
                }))
        else:
            instance = tf.train.Example(
                features=tf.train.Features(feature={
                    'x': _bytes_feature(im_string),
                    'y': _int64_feature(im_cla)
                }))
        writer.write(instance.SerializeToString())
        if image_index % 2000 == 0:
            sys.stdout.write('\r {}/{} instances finished.'.format(image_index + 1, num_images))
        if image_index % num_images_per_tfrecord == (num_images_per_tfrecord - 1):
            writer.close()

    writer.close()
    duration = time.time() - start_time
    sys.stdout.write('\n All {} instances finished in {:.1f} seconds'.format(num_images, duration))


########################################################################
class ReadTFRecords(object):
    def __init__(
            self, filenames, num_features=None, num_labels=0, x_dtype=tf.string, y_dtype=tf.int64, batch_size=16,
            skip_count=0, file_repeat=1, num_epoch=None, file_folder=None,
            num_threads=8, buffer_size=2000, shuffle_file=False,
            decode_jpeg=False, use_one_hot_label=False, use_smooth_label=True, num_classes=1):
        """ This function creates a dataset object that reads data from files.

        :param filenames: string or list of strings, e.g., 'train_000', ['train_000', 'train_001', ...]
        :param num_features: e.g., 3*299*299
        :param num_labels: 0 or positive integer, but the case for multiple labels is ambiguous if one_hot_label
            is to be used. Thus, we do not do that here.
        :param x_dtype: default tf.string, the dtype of features stored in tfrecord file
        :param y_dtype: default tf.int64, the dtype of labels stored in tfrecord file
        :param num_epoch: integer or None
        :param buffer_size:
        :param batch_size: integer
        :param skip_count: if num_instance % batch_size != 0, we could skip some instances
        :param file_repeat: if num_instance % batch_size != 0, we could repeat the files for k times
        :param num_epoch:
        :param file_folder: if not specified, DEFAULT_IN_FILE_DIR is used.
        :param num_threads:
        :param buffer_size:
        :param shuffle_file: bool, whether to shuffle the filename list
        :param decode_jpeg: if input is saved as JPEG string, set this to true
        :param use_one_hot_label: whether to expand the label to one-hot vector
        :param use_smooth_label: if uses smooth label instead of 0 and 1, to prevent overfitting
        :param num_classes: if use_one_hot_label is true, the number of classes also needs to be provided.

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
        self.decode_jpeg = decode_jpeg
        self.use_one_hot_label = False if num_labels > 1 else use_one_hot_label
        self.use_smooth_label = use_smooth_label
        self.num_classes = num_classes
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
            if self.decode_jpeg:
                # first decode compressed image string to uint8, as data is stored in this way
                # datum['x'] = tf.image.decode_image(datum['x'], channels=3)
                datum['x'] = tf.image.decode_jpeg(datum['x'], channels=3)
            else:
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
            else:
                datum['y'] = tf.cast(datum['y'], self.y_dtype)
            if self.use_one_hot_label:
                datum['y'] = tf.reshape(tf.one_hot(datum['y'], self.num_classes), (-1, ))
                if self.use_smooth_label:  # label smoothing
                    datum['y'] = 0.9 * datum['y'] + 0.1 / self.num_classes
            return datum['x'], datum['y']
        else:
            return datum['x']

    ###################################################################
    def image_preprocessor(self, channels, height, width, resize=None, image_augment_fun=None):
        """ This function shapes the input instance to image tensor.

        :param channels:
        :param height:
        :param width:
        :param resize: list of tuple
        :type resize: list, tuple
        :param image_augment_fun: the function applied to augment a single image
        :return:
        """

        def __preprocessor__(image):
            # scale image to [0, 1] or [-1,1]
            image = tf.divide(image, 255.0, name='scale_range')
            # image = tf.subtract(tf.divide(image, 127.5), 1, name='scale_range')

            # reshape - note this is determined by how the data is stored in tfrecords, modify with caution
            if self.decode_jpeg:
                # if decode_jpeg is true, the image is already in [height, width, channels] format,
                # thus we only need to consider IMAGE_FORMAT
                if FLAGS.IMAGE_FORMAT == 'channels_first':
                    image = tf.transpose(image, perm=(2, 0, 1))
                pass
            else:
                # if decode_jpeg is false, the image is probably in vector format
                # thus we reshape the image according to the stored format provided by IMAGE_FORMAT
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

            # apply augmentation method if provided
            if image_augment_fun is not None:
                image = image_augment_fun(image)

            return image

        # do image pre-processing
        if self.num_labels == 0:
            self.dataset = self.dataset.map(
                lambda image_data: __preprocessor__(image_data),
                num_parallel_calls=self.num_threads)
        else:
            self.dataset = self.dataset.map(
                lambda image_data, label: (__preprocessor__(image_data), label),
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
            # shuffle and repeat
            print(
                'The dataset repeats for infinite number of epochs' if self.num_epoch in {-1, None}
                else 'The dataset repeat {} epochs'.format(self.num_epoch))
            if shuffle_data:
                self.dataset = self.dataset.shuffle(self.buffer_size)
            #     self.dataset = self.dataset.apply(
            #         tf.data.experimental.shuffle_and_repeat(self.buffer_size, self.num_epoch))
            # else:
            #     self.dataset = self.dataset.repeat(self.num_epoch)
            self.dataset = self.dataset.repeat(self.num_epoch)
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
                    print(
                        'Caution: samples from class {}. This should not be used in training'.format(sample_class))
                    self.dataset = self.dataset.filter(lambda x, y: tf.equal(y[0], sample_class))
                    self.dataset = self.dataset.batch(self.batch_size)
            else:
                self.dataset = self.dataset.batch(self.batch_size)
            # self.dataset = self.dataset.padded_batch(batch_size)
            # prefetch to speed up input pipeline. After batch method, buffer_size in prefetch means one batch
            self.dataset = self.dataset.prefetch(buffer_size=1)

            self.scheduled = True

    ###################################################################
    def next_batch(self, sample_same_class=False, sample_class=None, shuffle_data=True):
        """ This function generates next batch

        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class are sampled.
            The sample_class is compared against the first label in num_labels.
            Note that sample_class should be smaller than the num of classes in total.
            Note that class_label should not be provided during training.
        :param shuffle_data:
        :return:
        """
        if self.num_labels == 0:
            if not self.scheduled:
                self.scheduler(shuffle_data=shuffle_data)
            if self.iterator is None:
                self.iterator = self.dataset.make_one_shot_iterator()

            x_batch = self.iterator.get_next()
            x_batch.set_shape(self.batch_shape)

            return {'x': x_batch}
        else:
            if sample_class is not None:
                assert isinstance(sample_class, (np.integer, int)), \
                    'class_label must be integer.'
                sample_same_class = True
            if not self.scheduled:
                self.scheduler(
                    shuffle_data=shuffle_data, sample_same_class=sample_same_class,
                    sample_class=sample_class)
            if self.iterator is None:
                self.iterator = self.dataset.make_one_shot_iterator()

            x_batch, y_batch = self.iterator.get_next()
            x_batch.set_shape(self.batch_shape)
            if self.use_one_hot_label:
                y_batch.set_shape([self.batch_size, self.num_classes])
            else:
                y_batch.set_shape([self.batch_size, self.num_labels])

            return {'x': x_batch, 'y': y_batch}


########################################################################
def read_inaturalist(key='train', batch_size=64, target_size=299, do_augment=False, buffer_size=2000, fast_mode=False):
    """ This function reads the iNaturalist 2019 dataset.

    :param key: train, val or test
    :param batch_size:
    :param target_size: the image size
    :param do_augment: True or false
    :param buffer_size:
    :return:
    """

    data_size = {'train': 265213, 'val': 3030, 'test': 35350}
    data_label = {'train': 1, 'val': 1, 'test': 0}
    num_images = data_size[key]
    steps_per_epoch = num_images // batch_size
    skip_count = num_images % batch_size
    num_labels = data_label[key]
    num_classes = 1010

    filenames = os.listdir(FLAGS.DEFAULT_IN)
    filenames = [filename.replace('.tfrecords', '') for filename in filenames if key in filename]
    print('The following tfrecords are read: {}'.format(filenames))

    dataset = ReadTFRecords(
        filenames, num_labels=num_labels, batch_size=batch_size, buffer_size=buffer_size,
        skip_count=skip_count, num_threads=8, decode_jpeg=True,
        use_one_hot_label=True, use_smooth_label=True if key == 'train' else False, num_classes=num_classes)
    if do_augment:
        from GeneralTools.inception_preprocessing import preprocess_image
        # apply basic data augmentation (random crops, random left-right flipping, color distortion)
        dataset.image_preprocessor(
            3, target_size, target_size,
            lambda x: preprocess_image(
                x, height=target_size, width=target_size,
                is_training=True if key == 'train' else False, fast_mode=fast_mode))
    else:
        dataset.image_preprocessor(3, target_size, target_size)
    dataset.scheduler(shuffle_data=False if key == 'test' else True)

    return dataset, steps_per_epoch
