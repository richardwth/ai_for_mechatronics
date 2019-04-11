""" This code builds a classifier.

"""
# default modules
import numpy as np
import tensorflow as tf

from GeneralTools.my_graph import graph_configure, MySession, prepare_folder
from GeneralTools.my_input import DatasetFromTensor
from GeneralTools.my_layer import SequentialNet


class MyClassifier(object):
    def __init__(
            self, architecture, input_shape, num_class, loss_type='cross-entropy',
            optimizer='adam', do_summary=True, name='classifier'):
        """ This function initializes a classification model

        :param architecture: a list of dictionary
        :param input_shape:
        :param num_class:
        :param loss_type:
        :param optimizer:
        :param do_summary:
        :param kwargs:
        """
        # structure parameters
        self.architecture = architecture
        self.loss_type = loss_type
        self.optimizer = optimizer
        # control parameters
        self.name = name
        self.do_summary = do_summary
        self.global_step = None
        # model parameter
        self.num_class = num_class
        # if last layer uses nonlinear activation function, ignore it
        # this is because the cross entropy loss function requires logits
        if architecture[-1].get('act', 'linear') is not 'linear':
            self.last_layer_act = architecture[-1]['act']
            architecture[-1]['act'] = 'linear'
        else:
            self.last_layer_act = None
        if num_class == 2:
            assert architecture[-1]['out'] == 1, \
                '{}: For two class problem, the last layer has one neuron.'.format(self.name)
        elif num_class > 2:
            assert architecture[-1]['out'] == num_class, \
                '{}: For {}-class problem, the last layer must have has {} neurons.'.format(
                    self.name, self.num_class, self.num_class)
        else:
            raise AttributeError('{}: The number of classes cannot be {}'.format(self.name, self.num_class))
        # initialize the network
        self.mdl = SequentialNet(self.architecture, input_shape=input_shape, name=self.name)

    def check_inputs(self, x, y):
        """ x and y are numpy ndarray

        :param x:
        :param y:
        :return:
        """
        assert isinstance(x, np.ndarray), \
            '{}: Input x must be numpy ndarray.'.format(self.name)
        assert isinstance(y, np.ndarray), \
            '{}: Input y must be numpy ndarray.'.format(self.name)
        assert x.shape[0] == y.shape[0], \
            '{}: Shape[0] of x and y must equal, but got {} and {}.'.format(
                self.name, x.shape[0], y.shape[0])

    def build_dataset(
            self, x=None, y=None, sample_mode='batch', batch_size=None, map_func=None, shuffle=True, name=None):
        """ This function checks the input data and build a dataset

        :param x:
        :param y:
        :param sample_mode: 'batch' or 'all'
        :param batch_size:
        :param map_func:
        :param shuffle:
        :param name:
        :return:
        """
        if x is not None and y is not None:
            # check inputs
            self.check_inputs(x, y)
            # if sample_mode is 'batch', create a random batch
            if sample_mode.lower() in {'batch'}:
                dataset = DatasetFromTensor(
                    (x, y), (tf.float32, tf.int32),
                    map_func=map_func,  # e.g., lambda x, y: (x/255.0, y)
                    name=name).schedule(batch_size, shuffle=shuffle)
                x_batch, y_batch = dataset.next()
                init_bundle = dataset.initializer()
            # if sample_mode is 'all', use all data
            elif sample_mode.lower() in {'all'}:
                x_batch = x
                y_batch = y
                init_bundle = None
            else:
                raise NotImplementedError(
                    '{}: The sample mode {} is not implemented.'.format(
                        self.name, sample_mode))
        elif x is not None or y is not None:
            raise AttributeError(
                '{}: If x or y is provided, the other one must be provided.'.format(self.name))
        else:
            x_batch = None
            y_batch = None
            init_bundle = None

        return x_batch, y_batch, init_bundle

    def cal_loss(self, target=None, prediction=None, name='loss'):
        """ This function calculates the loss function between targets and predictions

        :param target: [N, ] numpy array or tf.Tensor
        :param prediction: [N, d] tf.Tensor, d = 1 for two-class problem, d = num_class for multi-class problem
        :param name:
        :return:
        """
        if target is None:
            return None, None
        if prediction is None:
            raise AttributeError('{}: The prediction is not given.'.format(self.name))

        with tf.name_scope(name):
            accuracy = tf.metrics.accuracy(target, tf.argmax(prediction, axis=1), name='acc')
            if self.num_class == 2:
                target_float = tf.cast(tf.expand_dims(target, axis=1), dtype=tf.float32)  # [N, 1]
                if self.loss_type.lower() in {'cross-entropy', 'cross entropy'}:
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=target_float, logits=prediction, name='cross-entropy')
                    tf.losses.add_loss(loss)
                elif self.loss_type.lower() in {'hinge'}:
                    loss = tf.losses.hinge_loss(labels=target_float, logits=prediction)
                else:
                    raise NotImplementedError(
                        '{}: Loss type {} is not implemented for two-class problem.'.format(
                            self.name, self.loss_type))
            else:
                if self.loss_type.lower() in {'cross-entropy', 'cross entropy'}:
                    loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=prediction)
                else:
                    raise NotImplementedError(
                        '{}: Loss type {} is not implemented for multi-class problem.'.format(
                            self.name, self.loss_type))

        return loss, accuracy

    def train(self, x_train, y_train, batch_size=100, map_func=None,
              x_validate=None, y_validate=None, validation_mode='batch', validation_batch_size=100,
              init_learning_rate=0.01, decay_steps=None, optimizer=None,
              do_save=True, load_ckpt=False, model_folder='', max_step=1000, query_step=500,
              debug_mode=False, do_trace=False):
        """

        :param x_train: [N, D] numpy ndarray
        :param y_train: [N, ] numpy ndarray, expected value 0 or 1
        :param batch_size: scalar
        :param map_func: a function that maps two inputs (x, y) to two outputs (x, y), e.g., lambda x, y: (x/255.0, y)
        :param x_validate: [N, D] numpy ndarray
        :param y_validate: [N, ] numpy ndarray
        :param validation_mode: if validation set is too large, e.g., 1000, set validation mode to 'batch';
            otherwise, 'all'. 'batch' model will randomly sample a batch of validation samples.
        :param validation_batch_size:
        :param init_learning_rate:
        :param decay_steps:
        :param optimizer:
        :param do_save: save the model after training
        :param load_ckpt: load from previously saved model
        :param model_folder: folder for saving, visualising and restoring model
        :param max_step: max step for training
        :param query_step:
        :param debug_mode:
        :param do_trace:
        :return:
        """
        # build training and validation datasets
        num_samples = x_train.shape[0]
        step_per_epoch = num_samples // batch_size
        xtr, ytr, train_init_bundle = self.build_dataset(
            x_train, y_train, 'batch', batch_size, map_func, name='TrainData')
        xva, yva, validate_init_bundle = self.build_dataset(
            x_validate, y_validate, validation_mode, validation_batch_size, map_func, name='ValidateData')

        # make predictions
        ytr_prediction = self.mdl(xtr, training=True)
        if xva is None:
            yva_prediction = None
        else:
            yva_prediction = self.mdl(yva, training=False)

        # calculate loss
        loss_tr, acc_tr = self.cal_loss(ytr, ytr_prediction)
        loss_va, acc_va = self.cal_loss(yva, yva_prediction)

        # configure optimizer
        global_step, opt_op, _ = graph_configure(
            init_learning_rate, lr_decay_steps=decay_steps,
            optimizer=self.optimizer if optimizer is None else optimizer)
        # train_op = opt_op.minimize(loss_tr, global_step=global_step)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)
        grads_list = opt_op.compute_gradients(loss_tr, var_list)
        train_op = opt_op.apply_gradients(grads_list, global_step=global_step)

        # set summary
        if self.do_summary:
            tf.summary.scalar('loss/loss_tr', loss_tr)
            tf.summary.scalar('loss/acc_tr', acc_tr)
            if loss_va is not None:
                tf.summary.scalar('loss/loss_va', loss_va)
                tf.summary.scalar('loss/acc_va', acc_va)
            # add gradients to summary
            for var_grad, var in grads_list:
                var_name = var.name.replace(':', '_')
                tf.summary.histogram('grad_' + var_name, var_grad)
                tf.summary.histogram(var_name, var)
            summary_op = tf.summary.merge_all()
        else:
            summary_op = None
        ckpt_folder, summary_folder, save_path = prepare_folder(model_folder)

        # call the session
        if loss_va is None:
            query_list = None
            query_message = 'training loss: {}, training accuracy: {}'
        else:
            query_list = [loss_tr, acc_tr, loss_va, acc_va]
            query_message = 'training loss: {}, accuracy: {}, validation loss: {}, acc: {}'
        if debug_mode:
            with MySession(do_save=do_save, do_trace=do_trace, load_ckpt=load_ckpt) as sess:
                sess.run(train_init_bundle[0], feed_dict=train_init_bundle[1])
                sess.debug_mode(
                    train_op, [loss_tr, acc_tr], global_step,
                    summary_op, summary_folder=summary_folder, ckpt_folder=ckpt_folder, save_path=save_path,
                    max_step=max_step, query_list=query_list, query_step=query_step, query_message=query_message)
        else:
            with MySession(do_save=do_save, load_ckpt=load_ckpt) as sess:
                sess.run(train_init_bundle[0], feed_dict=train_init_bundle[1])
                sess.full_run(
                    train_op, [loss_tr, acc_tr], max_step, step_per_epoch, global_step,
                    summary_op, summary_folder=summary_folder, ckpt_folder=ckpt_folder, save_path=save_path,
                    query_list=query_list, query_step=query_step, query_message=query_message)

    def test(self, x_test, y_test, test_mode='batch', batch_size=100, map_func=None, load_ckpt=True, model_folder=''):
        """ This function test the model on test data

        :param x_test: [N, D] numpy ndarray
        :param y_test: [N, ] numpy ndarray, expected value 0 or 1
        :param test_mode: if test set is too large, e.g., 1000, set test mode to 'batch';
            otherwise, 'all'. 'batch' model will sequentially sample a batch of test samples.
        :param batch_size:
        :param map_func:
        :param load_ckpt:
        :param model_folder:
        :return:
        """
        # build training and validation datasets
        num_samples = x_test.shape[0]
        step_per_epoch = num_samples // batch_size if test_mode.lower() is 'batch' else 1
        xte, yte, test_init_bundle = self.build_dataset(
            x_test, y_test, test_mode, batch_size, map_func, shuffle=False, name='TestData')
        if xte is None:
            raise AttributeError('{}: Test dataset must be given.'.format(self.name))

        # make predictions
        yte_prediction = self.mdl(xte, training=False)

        # calculate loss
        loss_te, acc_te = self.cal_loss(yte, yte_prediction)

        # call the session
        ckpt_folder, summary_folder, save_path = prepare_folder(model_folder)
        with MySession(load_ckpt=load_ckpt) as sess:
            sess.run(test_init_bundle[0], feed_dict=test_init_bundle[1])
            value_list = sess.run_m_times(
                [yte_prediction, loss_te, acc_te],
                ckpt_folder=ckpt_folder, max_iter=step_per_epoch, keep_m_outputs=True)

        # prepare for output
        predictions = np.concatenate([values[0] for values in value_list], axis=0)
        loss = np.mean([values[1] for values in value_list])
        acc = np.mean([values[2] for values in value_list])

        return predictions, loss, acc
