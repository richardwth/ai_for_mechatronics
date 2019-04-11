"""
This file contains functions and classes to train a model
"""
import tensorflow as tf
import numpy as np
from GeneralTools.misc_fun import FLAGS


########################################################################
def weight_initializer(act_fun, init_w_scale=1.0):
    """ This function includes initializer for several common activation functions.
    The initializer will be passed to tf.layers.

    Notes
    FAN_AVG: for tensor shape (a, b, c), fan_avg = a * (b + c) / 2

    :param act_fun:
    :param init_w_scale:
    :return:
    """

    if FLAGS.WEIGHT_INITIALIZER == 'default':
        if init_w_scale == 0.0:
            initializer = tf.zeros_initializer()
        else:
            if act_fun == 'relu':
                initializer = tf.variance_scaling_initializer(
                    scale=2.0 * init_w_scale, mode='fan_in', distribution='normal')
            elif act_fun == 'lrelu':  # assume alpha = 0.1
                initializer = tf.variance_scaling_initializer(
                    scale=2.0 / 1.01 * init_w_scale, mode='fan_in', distribution='normal')
            elif act_fun == 'sigmoid':
                initializer = tf.variance_scaling_initializer(
                    scale=16.0 * init_w_scale, mode='fan_avg', distribution='uniform')
            else:  # xavier initializer
                initializer = tf.variance_scaling_initializer(
                    scale=1.0 * init_w_scale, mode='fan_avg', distribution='uniform')
    elif FLAGS.WEIGHT_INITIALIZER == 'sn_paper':
        # paper on spectral normalization used truncated_normal_initializer
        if FLAGS.VERBOSE:
            print('You are using custom initializer.')
        initializer = tf.truncated_normal_initializer(stddev=0.02)
    elif FLAGS.WEIGHT_INITIALIZER == 'pg_paper':
        # paper on progressively growing gan adjust the weight on runtime.
        if FLAGS.VERBOSE:
            print('You are using custom initializer.')
        initializer = tf.truncated_normal_initializer()
    else:
        raise NotImplementedError('The initializer {} is not implemented.'.format(FLAGS.WEIGHT_INITIALIZER))

    return initializer


########################################################################
def bias_initializer(init_b_scale=0.0):
    """ This function includes initializer for bias

    :param init_b_scale:
    :return:
    """
    if init_b_scale == 0.0:
        initializer = tf.zeros_initializer()
    else:  # a very small initial bias can avoid the zero outputs that may cause problems
        initializer = tf.truncated_normal_initializer(stddev=init_b_scale)

    return initializer


########################################################################
def leaky_relu(features, name=None):
    """ This function defines leaky rectifier linear unit

    :param features:
    :param name:
    :return:
    """
    # return tf.maximum(tf.multiply(0.1, features), features, name=name)
    return tf.nn.leaky_relu(features, alpha=0.1, name=name)


########################################################################
def get_std_act_fun(act_fun_name):
    """ This function gets the standard activation function from TensorFlow

    :param act_fun_name:
    :return:
    """
    if act_fun_name == 'linear':
        act_fun = tf.identity
    elif act_fun_name == 'relu':
        act_fun = tf.nn.relu
    elif act_fun_name == 'crelu':
        act_fun = tf.nn.crelu
    elif act_fun_name == 'elu':
        act_fun = tf.nn.elu
    elif act_fun_name == 'lrelu':
        act_fun = leaky_relu
    elif act_fun_name == 'selu':
        act_fun = tf.nn.selu
    elif act_fun_name == 'softplus':
        act_fun = tf.nn.softplus
    elif act_fun_name == 'softsign':
        act_fun = tf.nn.softsign
    elif act_fun_name == 'sigmoid':
        act_fun = tf.nn.sigmoid
    elif act_fun_name == 'tanh':
        act_fun = tf.nn.tanh
    elif act_fun_name == 'crelu':
        # concatenated ReLU doubles the depth of the activations
        # CReLU only supports NHWC
        act_fun = tf.nn.crelu
    elif act_fun_name == 'elu':
        act_fun = tf.nn.elu
    else:
        raise NotImplementedError('Function {} is not implemented.'.format(act_fun_name))

    return act_fun


########################################################################
def apply_activation(logits, act_fun, name=None):
    """ This function applies element-wise activation function

    Inputs:
    :param logits: inputs to activation function
    :param act_fun: name of activation function
    :param name:

    """
    if isinstance(act_fun, str):
        act_fun = get_std_act_fun(act_fun)
    layer_output = act_fun(logits, name=name)

    return layer_output


########################################################################
def update_layer_design(layer_design):
    """ This function reads layer_design and outputs an universal layer design dictionary

    This function is only part of an example! Most of its applications indicated by the
    key-value pairs are not implemented here.

    :param layer_design:
    layer_design may have following keys: (most of the below keys are not implemented in this example)
        'name': scope of the layer
        'type':
            'default' - default layer layout: linear mapping, nonlinear activation;
            'res' - residual block with conv shortcut;
            'res_i' - residual block with identity shortcut;
            'dense' - densely connected convolutional neural networks
        'op':
            'c' - convolution layer;
            'd' - dense layer;
            'dcd' - dense + conditional dense;
            'dck' - dense + conditional scalar weight;
            'sc' - separable convolution layer;
            'i' - identity layer;
            'tc' - transpose convolution;
            'avg', 'max', 'sum' - mean-pool, max-pool and sum-pool;
            'cck', - convolution + conditional scalar weight;
            'tcck' - transposed + conditional scalar weight;
            or a list of keys, e.g., {'c', 'c', 'cck'}
        'out': number of output channels or features
        'bias': if bias should be used. When 'w_nm' is 'b', bias is not used
        'act': activation function
        'act_nm': activation normalization method.
            'lrn' - local response normalization
            'b', 'bn', 'BN' - batch normalization;
            'cbn', 'CBN' - conditional batch normalization
        'act_k': activation multiplier. When 'w_nm' is 's', the user can choose a multiply the activation with a
            constant to reimburse the norm loss caused by the activation.
        'w_nm': kernel normalization method.
            's' - spectral normalization;
            'h' - he normalization;
            None - no normalization is used.
        'w_p': kernel penalization method.
            's' - spectral penalization;
            None - no layer-wise penalty is used.
        'kernel': kernel size for convolution layer; integer, or list/tuple of integers for multiple conv ops
        'strides': strides for convolution layer; integer, or list/tuple of integers for multiple conv ops
        'dilation': dilation for convolution layer; integer, or list/tuple of integers for multiple conv ops
        'padding': 'SAME' or 'VALID'; padding for convolution layer; string, or list/tuple of strings for
            multiple convolution ops
        'scale': a list containing the method used for up-sampling and the scale factor;
            a positive scale factor means up-sampling, a negative means down-sampling
            None: do not apply scaling
            'ps': periodic shuffling, the factor can only be int
            'bil': bilinear sampling
            'bic': bicubic sampling
            'avg' or 'max': average pool - can only be used in down-sampling
        'in_reshape': a shape list, reshape the input before passing it to kernel
        'out_reshape': a shape list, reshape the output before passing it to next layer
        'aux': auxiliary values and commands
    :return:
    """
    template = {'name': None, 'type': 'default', 'op': 'd', 'out': None, 'bias': True,
                'act': 'linear', 'act_nm': None, 'act_k': False,
                'w_nm': None, 'w_p': None,
                'kernel': 3, 'strides': 1, 'dilation': 1, 'padding': 'SAME', 'scale': None,
                'in_reshape': None, 'out_reshape': None, 'aux': None}
    # update template with parameters from layer_design
    for key in layer_design:
        template[key] = layer_design[key]

    # check template values to avoid error in implementing algorithms
    # if (template['act_nm'] in {'bn', 'BN', 'cbn', 'CBN'}) and (template['act'] == 'linear'):
    #     template['act_nm'] = None  # batch normalization is not used with linear activation
    if template['act_nm'] in {'bn', 'BN'} and template['bias'] in {'b', 'bias'}:
        template['bias'] = False  # batch normalization is not used with common bias, but conditional bias may be used
    if template['act_nm'] in {'cbn', 'CBN'}:
        template['bias'] = False  # conditional batch normalization is not used with any bias
    if template['op'] in {'tc'}:
        # transpose conv is usually used as upsampling method
        template['scale'] = None
    # if template['w_nm'] is not None:  # weight normalization and act normalization cannot be used together
    #     template['act_nm'] = None
    if template['scale'] is not None:
        assert isinstance(template['scale'], (list, tuple)), \
            'Value for key "scale" must be list or tuple.'
    if template['w_nm'] is not None:  # This is because different normalization methods do not work in the same layer
        assert not isinstance(template['w_nm'], (list, tuple)), \
            'Value for key "w_nm" must not be list or tuple.'
    if isinstance(template['in_reshape'], tuple):
        template['in_reshape'] = list(template['in_reshape'])
    if isinstance(template['out_reshape'], tuple):
        template['out_reshape'] = list(template['out_reshape'])

    # output template
    if template['op'] in {'d', 'dcd', 'dck'}:
        return {key: template[key]
                for key in ['name', 'op', 'type', 'out', 'bias',
                            'act', 'act_nm', 'act_k',
                            'w_nm', 'w_p',
                            'in_reshape', 'out_reshape', 'aux']}
    elif template['op'] in ['sc', 'c', 'tc', 'avg', 'max', 'sum', 'cck', 'tcck']:
        return {key: template[key]
                for key in ['name', 'op', 'type', 'out', 'bias',
                            'act', 'act_nm', 'act_k',
                            'w_nm', 'w_p',
                            'kernel', 'strides', 'dilation', 'padding', 'scale',
                            'in_reshape', 'out_reshape', 'aux']}
    elif template['op'] in {'i'}:
        return {key: template[key]
                for key in ['name', 'op', 'act', 'act_nm', 'type', 'in_reshape', 'out_reshape']}
    else:
        raise AttributeError('layer op {} not supported.'.format(template['op']))


########################################################################
class Layer(object):
    def __init__(self, design, input_shape=None, name_prefix=''):
        # layer definition as dictionary
        self.design = update_layer_design(design)
        # scope
        self.layer_scope = name_prefix + self.design['name']
        # IO
        self._layer_output_ = None
        self.input_shape = list(input_shape) if isinstance(input_shape, tuple) else input_shape
        self.output_shape = None
        # layer status
        self.is_layer_built = False
        # ops
        self.ops = {}

    def _input_(self, layer_input):
        """ This function initializes layer_output.

        :param layer_input: a tf.Tensor
        :return:
        """
        # check input shape
        input_shape = layer_input.get_shape().as_list()
        if self.input_shape is None:
            self.input_shape = input_shape
        else:
            assert self.input_shape[1:] == input_shape[1:], \
                '{}: the actual input shape {} does not match theoretic shape {}.'.format(
                    self.layer_scope, input_shape[1:], self.input_shape[1:])

        # copy input
        # It should be noted that we do not want any ops in current layer to change the layer_input in memory
        # This is because the layer_input may be fed into other layers.
        # However, if we use self._layer_output_ = layer_input directly, ops like partial assignment
        # self._layer_output_[:, 1] = self._layer_output_[:, 1] + 0.1 will change layer_input
        self._layer_output_ = tf.identity(layer_input)

    def _output_(self):
        """ This function returns the output
        :return layer_output: a tf.Tensor
        """
        output_shape = self._layer_output_.get_shape().as_list()
        if self.output_shape is None:
            self.output_shape = output_shape
        else:
            assert self.output_shape[1:] == output_shape[1:], \
                '{}: the actual output shape {} does not match theoretic shape {}.'.format(
                    self.layer_scope, output_shape[1:], self.output_shape[1:])
        # Layer object always forgets self._layer_output_ after its value is returned
        layer_output = self._layer_output_
        self._layer_output_ = None
        return layer_output

    def _add_kernel_(self, input_shape, name_scope='kernel'):
        """

        :param input_shape:
        :param name_scope:
        :return:
        """
        weight_init = weight_initializer(self.design['act'])
        if self.design['op'] == 'd':
            self.ops[name_scope] = tf.layers.Dense(
                self.design['out'],
                use_bias=self.design['bias'], kernel_initializer=weight_init,
                name=None)
            output_shape = [input_shape[0], self.design['out']]
        elif self.design['op'] == 'c':
            self.ops[name_scope] = tf.layers.Conv2D(
                self.design['out'],
                kernel_size=self.design['kernel'], strides=self.design['strides'],
                dilation_rate=self.design['dilation'], padding=self.design['padding'],
                data_format=FLAGS.IMAGE_FORMAT,
                use_bias=self.design['bias'], kernel_initializer=weight_init,
                name=self.layer_scope)
            # calculate output shape
            if FLAGS.IMAGE_FORMAT == 'channels_first':
                h, w = input_shape[2:]
            else:
                h, w = input_shape[1:3]
            h, w = spatial_shape_after_conv(
                [h, w], self.design['kernel'], self.design['strides'], self.design['dilation'],
                self.design['padding'])
            output_shape = [self.input_shape[0], self.design['out'], h, w] \
                if FLAGS.IMAGE_FORMAT == 'channels_first' else [input_shape, h, w, self.design['out']]
        else:
            raise NotImplementedError(
                '{}: The kernel operation {} is not implemented.'.format(
                    self.layer_scope, self.design['op']))
        return output_shape

    def _add_bn_(self, name_scope='BN'):
        if self.design['act_nm'] in {'bn', 'BN'}:
            axis = 1 if FLAGS.IMAGE_FORMAT == 'channels_first' else -1
            self.ops[name_scope] = tf.layers.BatchNormalization(axis=axis, fused=True)

    def _add_layer_default_(self, input_shape):
        """ This function adds the default layer with the following order of operations:
        upsampling - kernel - bias - BN - downsampling

        :param input_shape:
        :return:
        """
        # upsampling
        if self.design.get('scale') is not None:
            if self.design['scale'][1] > 0:  # upsampling
                pass
        # kernel + bias
        output_shape = self._add_kernel_(input_shape, 'kernel')
        # batch normalization
        if self.design['act_nm'] in {'bn', 'BN'}:
            self._add_bn_('BN')
        # activation
        # downsampling
        if self.design.get('scale') is not None:
            if self.design['scale'][1] < 0:  # downsampling
                pass

        return output_shape

    def _apply_layer_default_(self, training=True):
        """ This function applies the default layer

        The order of operations:
        upsampling - kernel - bias - BN - activation - downsampling

        :param training: for batch normalization, the statistics are calculated
        in different way during training and test.
        :return:
        """
        layer_out = self._layer_output_
        # upsampling
        if 'upsampling' in self.ops:
            pass
        # kernel + bias
        layer_out = self.ops['kernel'].apply(layer_out)
        # batch normalization
        if 'BN' in self.ops:
            layer_out = self.ops['BN'].apply(layer_out, training=training)
        # activation
        layer_out = self._apply_activation_(layer_out)
        # downsampling
        if 'downsampling' in self.ops:
            pass

        self._layer_output_ = layer_out

    def _apply_activation_(self, layer_input):
        """ This function applies activation function on the input

        :param layer_input:
        :return:
        """
        return apply_activation(layer_input, self.design['act'])

    def _apply_input_reshape_(self):
        if self.design['in_reshape'] is not None:
            batch_size = self._layer_output_.get_shape().as_list()[0]
            self._layer_output_ = tf.reshape(
                self._layer_output_, shape=[batch_size] + self.design['in_reshape'])

    def _apply_output_reshape_(self):
        if self.design['out_reshape'] is not None:
            batch_size = self._layer_output_.get_shape().as_list()[0]
            self._layer_output_ = tf.reshape(
                self._layer_output_, shape=[batch_size] + self.design['out_reshape'])

    def build_layer(self):
        """ This function builds the layer

        There are two static operations: in_reshape and out_reshape
        Other operations are defined by design['type']:
        'default': upsampling - kernel - bias - batch_norm - act - downsampling

        :return:
        """
        if not self.is_layer_built:
            # in case input is reshaped, the new shape is used
            if self.design['in_reshape'] is None:
                output_shape = self.input_shape
            else:
                output_shape = [self.input_shape[0]] + self.design['in_reshape']

            # register ops
            if self.design['type'].lower() in {'default'}:
                output_shape = self._add_layer_default_(output_shape)
            else:
                raise NotImplementedError(
                    '{}: {} is not implemented.'.format(self.layer_scope, self.design['type']))

            # in case output is reshaped, the new shape is used
            if self.design['out_reshape'] is None:
                self.output_shape = output_shape
            else:
                self.output_shape = [output_shape[0]] + self.design['out_reshape']

        self.is_layer_built = True

    def __call__(self, layer_input, training=True):
        """ This function calculates layer_output based on layer_input

        :param layer_input: a tensor x
        :param training:
        :return:
        """
        self.build_layer()  # in case layer has not been build
        with tf.variable_scope(self.layer_scope, reuse=tf.AUTO_REUSE):
            self._input_(layer_input)
            self._apply_input_reshape_()

            # register ops
            if self.design['type'].lower() in {'default'}:
                self._apply_layer_default_(training)

            self._apply_output_reshape_()
            return self._output_()

    def apply(self, layer_input, training=True):
        """ This function calls self.__call__

        :param layer_input: a tensor x
        :param training:
        :return:
        """
        return self.__call__(layer_input, training)


########################################################################
def spatial_shape_after_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after convolution layer.

    The formula is obtained from: https://www.tensorflow.org/api_docs/python/tf/nn/convolution

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
        if padding in ['same', 'SAME']:
            return np.int(np.ceil(input_spatial_shape / strides))
        else:
            return np.int(np.ceil((input_spatial_shape - (kernel_size - 1) * dilation) / strides))


########################################################################
def spatial_shape_after_transpose_conv(input_spatial_shape, kernel_size, strides, dilation, padding):
    """ This function calculates the spatial shape after convolution layer.

    Since transpose convolution is often used in upsampling, scale_factor is not used here.

    This function has not been fully tested, and may be wrong in some cases.

    :param input_spatial_shape:
    :param kernel_size:
    :param strides:
    :param dilation:
    :param padding:
    :return:
    """
    if isinstance(input_spatial_shape, (list, tuple)):
        return [spatial_shape_after_transpose_conv(
            one_shape, kernel_size, strides, dilation, padding) for one_shape in input_spatial_shape]
    else:
        if padding in ['same', 'SAME']:
            return np.int(input_spatial_shape * strides)
        else:
            return np.int(input_spatial_shape * strides + (kernel_size - 1) * dilation)


########################################################################
class SequentialNet(object):
    """ This class is designed to:
        1. ease construction of networks with complex structure
        2. apply layer-wise operations

    """

    def __init__(
            self, net_design, input_shape, name='net'):
        """ This function initializes a network

        :param net_design: a list of dictionaries
        :param input_shape: input shape to the first layer
        :param name:
        """
        # net definition
        self.net_def = net_design
        self.num_layers = len(net_design)
        # scope
        self.name_scope = name  # get the parent scope

        # initialize the layers
        self.layers = []
        output_shape = input_shape
        for i in range(self.num_layers):
            layer_design = self.net_def[i]
            self.layers.append(
                Layer(layer_design, input_shape=output_shape, name_prefix=self.name_scope + '/'))
            self.layers[i].build_layer()
            output_shape = self.layers[i].output_shape
            assert self.layers[i].output_shape is not None
        # self.layer_names = [layer.layer_scope for layer in self.layers]

    def __call__(self, net_input, training=True):
        net_output = net_input
        for layer in self.layers:
            net_output = layer.apply(net_output, training=training)

        return net_output

    def apply(self, net_input, training=True):
        return self.__call__(net_input, training)
