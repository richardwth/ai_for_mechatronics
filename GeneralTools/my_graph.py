"""
This file contains functions and classes to optimize a model
"""
import json
import os.path
import time
import warnings
import numpy as np
import tensorflow as tf
from GeneralTools.misc_fun import FLAGS
from tensorflow.python.client import timeline


########################################################################
def prepare_folder(filename, sub_folder='', set_folder=True):
    """ This function prepares the folders for summary and saving model

    :param filename:
    :param sub_folder:
    :param set_folder:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    ckpt_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_ckpt', sub_folder)
    if not os.path.exists(ckpt_folder) and set_folder:
        os.makedirs(ckpt_folder)
    summary_folder = os.path.join(FLAGS.DEFAULT_OUT, filename + '_log', sub_folder)
    if not os.path.exists(summary_folder) and set_folder:
        os.makedirs(summary_folder)
    save_path = os.path.join(ckpt_folder, filename + '.ckpt')

    return ckpt_folder, summary_folder, save_path


########################################################################
def prepare_embedding_folder(summary_folder, filename, file_index=''):
    """ This function prepares the files for embedding visualization

    :param summary_folder:
    :param filename:
    :param file_index:
    :return:
    """
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]

    embedding_path = os.path.join(summary_folder, filename + file_index + '_embedding.ckpt')
    label_path = os.path.join(summary_folder, filename + file_index + '_label.tsv')
    sprite_path = os.path.join(summary_folder, filename + file_index + '.png')

    return embedding_path, label_path, sprite_path


########################################################################
def write_metadata(label_path, labels, names=None):
    """ This function writes raw_labels to file for embedding

    :param label_path: file name, e.g. '...\\metadata.tsv'
    :param labels: raw_labels
    :param names: interpretation for raw_labels, e.g. ['plane','auto','bird','cat']
    :return:
    """
    metadata_file = open(label_path, 'w')
    metadata_file.write('Name\tClass\n')
    if names is None:
        i = 0
        for label in labels:
            metadata_file.write('%06d\t%s\n' % (i, str(label)))
            i = i + 1
    else:
        for label in labels:
            metadata_file.write(names[label])
    metadata_file.close()


########################################################################
def write_sprite(sprite_path, images, mesh_num=None, if_invert=False):
    """ This function writes images to sprite image for embedding

    This function was taken from:
    https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

    The input image must be channels_last format.

    :param sprite_path: file name, e.g. '...\\a_sprite.png'
    :param images: numpy ndarray, [batch_size, height, width(, channels)]
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param mesh_num: nums of images in the row and column, must be a tuple
    :return:
    """
    if len(images.shape) == 3:  # if dimension of image is 3, extend it to 4
        images = np.tile(images[..., np.newaxis], (1, 1, 1, 3))
    if images.shape[3] == 1:  # if last dimension is 1, extend it to 3
        images = np.tile(images, (1, 1, 1, 3))
    # scale image to range [0,1]
    images = images.astype(np.float32)
    image_min = np.min(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) - image_min).transpose((3, 0, 1, 2))
    image_max = np.max(images.reshape((images.shape[0], -1)), axis=1)
    images = (images.transpose((1, 2, 3, 0)) / image_max).transpose((3, 0, 1, 2))
    if if_invert:
        images = 1 - images
    # check mesh_num
    if mesh_num is None:
        if FLAGS.VERBOSE:
            print('Mesh_num will be calculated as sqrt of batch_size')
        batch_size = images.shape[0]
        sprite_size = int(np.ceil(np.sqrt(batch_size)))
        mesh_num = (sprite_size, sprite_size)
        # add paddings if batch_size is not the square of sprite_size
        padding = ((0, sprite_size ** 2 - batch_size), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
        images = np.pad(images, padding, mode='constant', constant_values=0)
    elif isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    # Tile the individual thumbnails into an image
    new_shape = mesh_num + images.shape[1:]
    images = images.reshape(new_shape).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((mesh_num[0] * images.shape[1], mesh_num[1] * images.shape[3]) + images.shape[4:])
    images = (images * 255).astype(np.uint8)
    # save images to file
    import imageio
    imageio.imwrite(sprite_path, images)


########################################################################
def write_sprite_wrapper(
        images, mesh_num, filename, file_folder=None, file_index='',
        if_invert=False, image_format='channels_last'):
    """ This is a wrapper function for write_sprite.

    :param images: numpy ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num: mus tbe tuple (row_mesh, column_mesh)
    :param filename:
    :param file_folder:
    :param file_index:
    :param if_invert: bool, if true, invert images: images = 1 - images
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if isinstance(mesh_num, list):
        mesh_num = tuple(mesh_num)
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    if image_format in {'channels_first', 'NCHW'}:  # convert to [batch_size, height, width, channels]
        images = np.transpose(images, axes=(0, 2, 3, 1))
    # set up file location
    sprite_path = os.path.join(file_folder, filename + file_index + '.png')
    # write to files
    if os.path.isfile(sprite_path):
        warnings.warn('This file already exists: ' + sprite_path)
    else:
        write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)


########################################################################
def embedding_latent_code(
        latent_code, file_folder, embedding_path, var_name='codes',
        label_path=None, sprite_path=None, image_size=None):
    """ This function visualize latent_code using tSNE or PCA. The results can be viewed
    on tensorboard.

    :param latent_code: 2-D data
    :param file_folder:
    :param embedding_path:
    :param var_name:
    :param label_path:
    :param sprite_path:
    :param image_size:
    :return:
    """
    # register a session
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    # prepare a embedding variable
    # note this must be a variable, not a tensor
    embedding_var = tf.Variable(latent_code, name=var_name)
    sess.run(embedding_var.initializer)

    # configure the embedding
    from tensorflow.contrib.tensorboard.plugins import projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # add metadata (label) to embedding; comment out if no metadata
    if label_path is not None:
        embedding.metadata_path = label_path
    # add sprite image to embedding; comment out if no sprites
    if sprite_path is not None:
        embedding.sprite.image_path = sprite_path
        embedding.sprite.single_image_dim.extend(image_size)
    # finalize embedding setting
    embedding_writer = tf.summary.FileWriter(file_folder)
    projector.visualize_embeddings(embedding_writer, config)
    embedding_saver = tf.train.Saver([embedding_var], max_to_keep=1)
    embedding_saver.save(sess, embedding_path)
    # close all
    sess.close()


########################################################################
def embedding_image_wrapper(
        latent_code, filename, var_name='codes', file_folder=None, file_index='',
        labels=None, images=None, mesh_num=None, if_invert=False, image_format='channels_last'):
    """ This function is a wrapper function for embedding_image

    :param latent_code: the data to visualise
    :param filename: the name used to prepare folder
    :param var_name: the name of the variable
    :param file_folder: specific file folder to save the visualization files
    :param file_index:
    :param labels:
    :param images: ndarray, [batch_size, height, width(, channels)], values in range [0,1]
    :param mesh_num:
    :param if_invert:
    :param image_format: the default is channels_last; if channels_first is provided, transpose will be done.
    :return:
    """
    # check inputs
    if not isinstance(filename, str):  # if list, use the name of the first file
        filename = filename[0]
    if file_folder is None:
        file_folder = FLAGS.DEFAULT_OUT
    # prepare folder
    embedding_path, label_path, sprite_path = prepare_embedding_folder(file_folder, filename, file_index)
    # write label to file if labels are given
    if labels is not None:
        if os.path.isfile(label_path):
            warnings.warn('Label file {} already exist.'.format(label_path))
        else:
            write_metadata(label_path, labels)
    else:
        label_path = None
    # write images to file if images are given
    if images is not None:
        # if image is in channels_first format, convert to channels_last format
        if image_format == 'channels_first':
            images = np.transpose(images, axes=(0, 2, 3, 1))
        image_size = images.shape[1:3]  # [height, width]
        if os.path.isfile(sprite_path):
            warnings.warn('Sprite file {} already exist.'.format(sprite_path))
        else:
            write_sprite(sprite_path, images, mesh_num=mesh_num, if_invert=if_invert)
    else:
        image_size = None
        sprite_path = None
    if os.path.isfile(embedding_path):
        warnings.warn('Embedding file {} already exist.'.format(embedding_path))
    else:
        embedding_latent_code(
            latent_code, file_folder, embedding_path, var_name=var_name,
            label_path=label_path, sprite_path=sprite_path, image_size=image_size)


########################################################################
def get_ckpt(ckpt_folder, ckpt_file=None):
    """ This function gets the ckpt states. In case an older ckpt file is needed, provide the name in ckpt_file

    :param ckpt_folder:
    :param ckpt_file:
    :return:
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_folder)
    if ckpt_file is None:
        return ckpt
    else:
        index_file = os.path.join(ckpt_folder, ckpt_file+'.index')
        if os.path.isfile(index_file):
            ckpt.model_checkpoint_path = os.path.join(ckpt_folder, ckpt_file)
        else:
            raise FileExistsError('{} does not exist.'.format(index_file))

        return ckpt


########################################################################
def print_tensor_in_ckpt(ckpt_folder, all_tensor_values=False, all_tensor_names=False):
    """ This function print the list of tensors in checkpoint file.

    Example:
    from GeneralTools.graph_func import print_tensor_in_ckpt
    ckpt_folder = '/home/richard/PycharmProjects/myNN/Results/cifar_ckpt/sngan_hinge_2e-4_nl'
    print_tensor_in_ckpt(ckpt_folder)

    :param ckpt_folder:
    :param all_tensor_values: Boolean indicating whether to print the values of all tensors.
    :param all_tensor_names: Boolean indicating whether to print all tensor names.
    :return:
    """
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    if not isinstance(ckpt_folder, str):  # if list, use the name of the first file
        ckpt_folder = ckpt_folder[0]

    output_folder = os.path.join(FLAGS.DEFAULT_OUT, ckpt_folder)
    print(output_folder)
    ckpt = tf.train.get_checkpoint_state(output_folder)
    print(ckpt)
    print_tensors_in_checkpoint_file(
        file_name=ckpt.model_checkpoint_path, tensor_name='',
        all_tensors=all_tensor_values, all_tensor_names=all_tensor_names)


########################################################################
def graph_configure(
        initial_lr, global_step_name='global_step', lr_decay_steps=None,
        end_lr=1e-7, optimizer='adam'):
    """ This function configures global_step and optimizer

    :param initial_lr:
    :param global_step_name:
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer:
    :return:
    """
    global_step = global_step_config(name=global_step_name)
    opt_op, learning_rate = opt_config(initial_lr, lr_decay_steps, end_lr, optimizer, global_step=global_step)

    return global_step, opt_op, learning_rate


########################################################################
def global_step_config(name='global_step'):
    """ This function is a wrapper for global step

    """
    global_step = tf.get_variable(
        name=name,
        shape=[],
        dtype=tf.int32,
        initializer=tf.constant_initializer(0),
        trainable=False)

    return global_step


########################################################################
def opt_config(
        initial_lr, lr_decay_steps=None, end_lr=1e-7,
        optimizer='adam', name_suffix='', global_step=None, target_step=1e5):
    """ This function configures optimizer.

    :param initial_lr:
    :param lr_decay_steps:
    :param end_lr:
    :param optimizer:
    :param name_suffix:
    :param global_step:
    :param target_step:
    :return:
    """
    if optimizer in ['SGD', 'sgd']:
        # sgd
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.GradientDescentOptimizer(
            learning_rate, name='GradientDescent'+name_suffix)
        if FLAGS.VERBOSE:
            print('GradientDescent Optimizer is used.')
    elif optimizer in ['Momentum', 'momentum']:
        # momentum
        if lr_decay_steps is None:
            lr_decay_steps = np.round(target_step * np.log(0.96) / np.log(end_lr / initial_lr)).astype(np.int32)
        learning_rate = tf.train.exponential_decay(  # adaptive learning rate
            initial_lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=0.96,
            staircase=False)
        opt_op = tf.train.MomentumOptimizer(
            learning_rate, momentum=0.9, name='Momentum'+name_suffix)
        if FLAGS.VERBOSE:
            print('Momentum Optimizer is used.')
    elif optimizer in ['Adam', 'adam']:  # adam
        # Occasionally, adam optimizer may cause the objective to become nan in the first few steps
        # This is because at initialization, the gradients may be very big. Setting beta1 and beta2
        # close to 1 may prevent this.
        learning_rate = tf.constant(initial_lr)
        # opt_op = tf.train.AdamOptimizer(
        #     learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8, name='Adam'+name_suffix)
        opt_op = tf.train.AdamOptimizer(
            learning_rate, beta1=0.5, beta2=0.999, epsilon=1e-8, name='Adam' + name_suffix)
        if FLAGS.VERBOSE:
            print('Adam Optimizer is used.')
    elif optimizer in ['RMSProp', 'rmsprop']:
        # RMSProp
        learning_rate = tf.constant(initial_lr)
        opt_op = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, name='RMSProp'+name_suffix)
        if FLAGS.VERBOSE:
            print('RMSProp Optimizer is used.')
    else:
        raise AttributeError('Optimizer {} not supported.'.format(optimizer))

    return opt_op, learning_rate


###################################################################
class TimeLiner:
    def __init__(self):
        """ This class creates a timeline object that can be used to trace the timeline of
            multiple steps when called at each step.

        """
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, trace_file):
        with open(trace_file, 'w') as f:
            json.dump(self._timeline_dict, f)


###################################################################
class MySession(object):
    def __init__(
            self, do_save=False, do_trace=False, save_path=None,
            load_ckpt=False, log_device=False, ckpt_var_list=None):
        """ This class provides shortcuts for running sessions.
        It needs to be run under context managers. Example:
        with MySession() as sess:
            var1_value, var2_value = sess.run_once([var1, var2])

        :param do_save: bool, whether to save the model
        :param do_trace: bool, whether to profile the model
        :param save_path: provide a model-level save_path to be used in full_run and debug_run
        :param load_ckpt: bool, whether to load previous saved model
        :param log_device:
        :param ckpt_var_list: list of variables to save / restore
        """
        # register a session
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device))
        # initialization
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = None
        self.threads = None
        if FLAGS.VERBOSE:
            print('Graph initialization finished...')
        # configuration
        self.ckpt_var_list = ckpt_var_list
        self.saver = self._get_saver_() if do_save else None
        self.save_path = save_path if do_save else None
        self.summary_writer = None
        self.do_trace = do_trace
        self.load_ckpt = load_ckpt

    def __enter__(self):
        """ The enter method is called when "with" statement is used.

        :return:
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ The exit method is called when leaving the scope of "with" statement

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        if FLAGS.VERBOSE:
            print('Session finished.')
        if self.summary_writer is not None:
            self.summary_writer.close()
        self.coord.request_stop()
        # self.coord.join(self.threads)
        self.sess.close()

    def _get_saver_(self):
        # create a saver to save all variables
        # Saver op should always be assigned to cpu, and it should be
        # created after all variables have been defined; otherwise, it
        # only save those variables already created.
        with tf.device('/cpu:0'):
            return tf.train.Saver(
                var_list=tf.global_variables() if self.ckpt_var_list is None else self.ckpt_var_list)

    def _load_ckpt_(self, ckpt_folder=None, ckpt_file=None, force_print=False):
        """ This function loads a checkpoint model

        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param force_print: whether to ignore FLAGS.VERBOSE or not
        :return:
        """
        if self.load_ckpt and (ckpt_folder is not None):
            ckpt = get_ckpt(ckpt_folder, ckpt_file=ckpt_file)
            if ckpt is None:
                if FLAGS.VERBOSE or force_print:
                    print('No ckpt Model found at {}. Model training from scratch.'.format(ckpt_folder))
            else:
                if self.saver is None:
                    self.saver = self._get_saver_()
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                if FLAGS.VERBOSE or force_print:
                    print('Model reloaded from {}.'.format(ckpt_folder))
        else:
            if FLAGS.VERBOSE or force_print:
                    print('No ckpt model is loaded for current calculation.')

    def _check_thread_(self):
        """ This function initializes the coordinator and threads

        :return:
        """
        if self.threads is None:
            self.coord = tf.train.Coordinator()
            # self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def run(self, var_list, ckpt_folder=None, ckpt_file=None, ckpt_var_list=None, feed_dict=None, do_time=False):
        """ This functions calculates var_list.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :param do_time:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()

        if do_time:
            start_time = time.time()
            var_value = self.sess.run(var_list, feed_dict=feed_dict)
            if FLAGS.VERBOSE:
                print('Running session took {:.3f} sec.'.format(time.time() - start_time))
        else:
            var_value = self.sess.run(var_list, feed_dict=feed_dict)

        return var_value

    def run_m_times(
            self, var_list, ckpt_folder=None, ckpt_file=None, max_iter=10000,
            keep_m_outputs=False, ckpt_var_list=None, feed_dict=None):
        """ This functions calculates var_list for multiple iterations, as often done in
        Monte Carlo analysis.

        :param var_list:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param max_iter:
        :param keep_m_outputs: whether to keep all outputs during iterations
        :param ckpt_var_list: the variable to load in order to calculate var_list
        :param feed_dict:
        :return:
        """
        if ckpt_var_list is not None:
            self.ckpt_var_list = ckpt_var_list
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file)
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        start_time = time.time()
        if keep_m_outputs:
            var_value_list = []
            for i in range(max_iter):
                var_value, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
                var_value_list.append(var_value)
        else:
            for i in range(max_iter - 1):
                _, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
            var_value_list, _ = self.sess.run([var_list, extra_update_ops], feed_dict=feed_dict)
        # global_step_value = self.sess.run([self.global_step])
        if FLAGS.VERBOSE:
            print('Calculation took {:.3f} sec.'.format(time.time() - start_time))
        return var_value_list

    @staticmethod
    def print_query(query_value_list, step=0, epoch=0, query_message=None):
        if query_message is None:
            print('Epoch {}, global steps {}, query value {}'.format(
                epoch, step,
                ['{}'.format(['<{:.2f}>'.format(l_val) for l_val in l_value])
                 if isinstance(l_value, (np.ndarray, list))
                 else '<{:.3f}>'.format(l_value)
                 for l_value in query_value_list]))
        else:
            message = 'Epoch {}, global steps {}, ' + query_message
            query_formatted = ['{}'.format(['<{:.2f}>'.format(l_val) for l_val in l_value])
                               if isinstance(l_value, (np.ndarray, list))
                               else '<{:.3f}>'.format(l_value)
                               for l_value in query_value_list]
            print(message.format(epoch, step, *query_formatted))

    def full_run(self, op_list, loss_list, max_step, step_per_epoch, global_step, summary_op=None,
                 summary_image_op=None, summary_folder=None, ckpt_folder=None, ckpt_file=None, save_path=None,
                 do_print_query=True, query_list=None, query_step=500, query_message=None, force_print=False):
        """ This function run the session with all monitor functions.

        :param op_list: the first op in op_list runs every extra_steps when the rest run once.
        :param loss_list: the first loss is used to monitor the convergence
        :param max_step:
        :param step_per_epoch:
        :param global_step:
        :param summary_op:
        :param summary_image_op: operations that add generated images to summary
        :param summary_folder:
        :param ckpt_folder: folder of ckpt file to load
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param save_path: if not provided, use self.save_path
        :param do_print_query:
        :param query_list: variable to monitor during query, if not provided, it is the loss_list
        :param query_step:
        :param query_message: if provided, use custom message
        :param force_print:
        :return:
        """
        # prepare input
        if not isinstance(loss_list, (list, tuple)):
            loss_list = [loss_list]
        # prepare writer
        if (summary_op is not None) or (summary_image_op is not None):
            self.summary_writer = tf.summary.FileWriter(summary_folder, self.sess.graph)
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file, force_print=force_print)
        # run the session
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # this is necessary for batch normalization
        start_time = time.time()
        for step in range(max_step):
            # update the model
            loss_value_list, _, _, global_step_value = self.sess.run(
                [loss_list, op_list, extra_update_ops, global_step])
            # check if model produces nan outcome
            assert not any(np.isnan(loss_value_list)), \
                'Model diverged with loss = {} at step {}'.format(loss_value_list, step)

            # add summary and print loss every query step
            if global_step_value % query_step == (query_step-1):
                if summary_op is not None:
                    summary_str = self.sess.run(summary_op)
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)
                if do_print_query:
                    epoch = step // step_per_epoch
                    if query_list is None:
                        self.print_query(loss_value_list, global_step_value, epoch, query_message)
                    else:
                        query_value_list = self.sess.run(query_list)
                        self.print_query(query_value_list, global_step_value, epoch, query_message)

            # save model at last step
            if step == max_step - 1:
                if self.saver is not None:
                    self.saver.save(
                        self.sess,
                        save_path=self.save_path if save_path is None else save_path,
                        global_step=global_step_value)
                if summary_image_op is not None:
                    summary_image_str = self.sess.run(summary_image_op)
                    self.summary_writer.add_summary(summary_image_str, global_step=global_step_value)

        # calculate sess duration
        duration = time.time() - start_time
        if FLAGS.VERBOSE:
            print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))

    def abnormal_save(self, loss_value_list, global_step_value, summary_op, save_path=None):
        """ This function save the model in abnormal cases

        :param loss_value_list:
        :param global_step_value:
        :param summary_op:
        :param save_path: if not provided, use self.save_path
        :return:
        """
        if any(np.isnan(loss_value_list)):
            # save the model
            if self.saver is not None:
                self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)
            warnings.warn('Training Stopped due to nan in loss: {}.'.format(loss_value_list))
            return True
        elif any(np.greater(loss_value_list, 30000)):
            # save the model
            if self.saver is not None:
                self.saver.save(
                        self.sess,
                        save_path=self.save_path if save_path is None else save_path,
                        global_step=global_step_value)
            # add summary
            if summary_op is not None:
                summary_str = self.sess.run(summary_op)
                self.summary_writer.add_summary(summary_str, global_step=global_step_value)
            warnings.warn('Training Stopped early as loss diverged.')
            return True
        else:
            return False

    def debug_mode(self, op_list, loss_list, global_step, summary_op=None, summary_folder=None, ckpt_folder=None,
                   ckpt_file=None, save_path=None, max_step=200,
                   do_print_query=True, query_list=None, query_step=500, query_message=None, force_print=False):
        """ This function debugs the code. It allows for tracing and saving in abnormal cases.

        :param op_list:
        :param loss_list:
        :param global_step:
        :param summary_op:
        :param summary_folder:
        :param max_step:
        :param ckpt_folder:
        :param ckpt_file: in case an older ckpt file is needed, provide it here, e.g. 'cifar.ckpt-6284'
        :param save_path:
        :param do_print_query:
        :param query_list:
        :param query_step:
        :param query_message: if provided, use custom message
        :param force_print:
        :return:
        """
        if self.do_trace or (summary_op is not None):
            self.summary_writer = tf.summary.FileWriter(summary_folder, self.sess.graph)
        if self.do_trace:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            multi_runs_timeline = TimeLiner()
        else:
            run_options = None
            run_metadata = None
            multi_runs_timeline = None
        if query_step > max_step:
            query_step = np.minimum(max_step-1, 100)

        # run the session
        self._load_ckpt_(ckpt_folder, ckpt_file=ckpt_file, force_print=force_print)
        self._check_thread_()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(extra_update_ops)
        start_time = time.time()
        for step in range(max_step):
            if self.do_trace and (step >= max_step - 5):
                # update the model in trace mode
                loss_value_list, _, global_step_value, _ = self.sess.run(
                    [loss_list, op_list, global_step, extra_update_ops],
                    options=run_options, run_metadata=run_metadata)
                # add time line
                self.summary_writer.add_run_metadata(
                    run_metadata, tag='step%d' % global_step_value, global_step=global_step_value)
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                chrome_trace = trace.generate_chrome_trace_format()
                multi_runs_timeline.update_timeline(chrome_trace)
            else:
                # update the model
                loss_value_list, _, global_step_value, _ = self.sess.run(
                    [loss_list, op_list, global_step, extra_update_ops])

            # print(loss_value_list) and add summary
            if global_step_value % query_step == 1:  # at step 0, global step = 1
                if do_print_query:
                    epoch = 0
                    if query_list is None:
                        self.print_query(loss_value_list, global_step_value, epoch, query_message)
                    else:
                        query_value_list = self.sess.run(query_list)
                        self.print_query(query_value_list, global_step_value, epoch, query_message)
                if summary_op is not None:
                    summary_str = self.sess.run(summary_op)
                    self.summary_writer.add_summary(summary_str, global_step=global_step_value)

            # in abnormal cases, save the model
            if self.abnormal_save(loss_value_list, global_step_value, summary_op, save_path):
                break

            # save the mdl if for loop completes normally
            if step == max_step - 1 and self.saver is not None:
                self.saver.save(self.sess, save_path=self.save_path, global_step=global_step_value)

        # calculate sess duration
        duration = time.time() - start_time
        print('Training for {} steps took {:.3f} sec.'.format(max_step, duration))
        # save tracing file
        if self.do_trace:
            trace_file = os.path.join(summary_folder, 'timeline.json')
            multi_runs_timeline.save(trace_file)
