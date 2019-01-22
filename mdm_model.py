import tensorflow as tf
import utils


class _DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 depthwise_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_DepthwiseConv2D, self).__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs)


def _depthwise_conv2d(inputs,
                      kernel_size,
                      strides=(1, 1),
                      padding='valid',
                      depth_multiplier=1,
                      data_format='channels_last',
                      activation=None,
                      use_bias=True,
                      depthwise_initializer=None,
                      bias_initializer=tf.zeros_initializer(),
                      depthwise_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      depthwise_constraint=None,
                      bias_constraint=None,
                      trainable=True,
                      name=None,
                      reuse=None):
    layer = _DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)


def _conv2d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        activation=None,
        use_bias=True,
        use_bn=False,
        training=False,
        name='Convolution'
):
    with tf.variable_scope(name, values=[inputs]):
        inputs = tf.layers.conv2d(
            inputs, filters, kernel_size, strides,
            padding='same',
            use_bias=use_bias,
            name='Conv2D'
        )
        if use_bn:
            inputs = tf.layers.batch_normalization(
                inputs,
                training=training,
                name='BatchNorm'
            )
        if activation is not None:
            inputs = activation(inputs)
    return inputs


def _conv2d_dw(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        activation=None,
        use_bias=True,
        use_bn=False,
        training=False,
        name='DepthWiseConvolution'
):
    with tf.variable_scope(name, values=[inputs]):
        inputs = _depthwise_conv2d(
            inputs, kernel_size, strides,
            padding='same',
            use_bias=use_bias,
            name='DWConv2D'
        )
        if use_bn:
            inputs = tf.layers.batch_normalization(
                inputs,
                training=training,
                name='BatchNorm1'
            )
        inputs = tf.layers.conv2d(
            inputs, filters, [1, 1],
            padding='same',
            use_bias=use_bias,
            name='Conv2D'
        )
        if use_bn:
            inputs = tf.layers.batch_normalization(
                inputs,
                training=training,
                name='BatchNorm2'
            )
        if activation is not None:
            inputs = activation(inputs)
    return inputs


def _shuffle_block(
        inputs,
        in_filters,
        out_filters,
        kernel_size,
        strides,
        depth,
        training=False,
        name='ShuffleBlock'
):
    with tf.variable_scope(name, values=[inputs]):
        with tf.variable_scope('Unit0'):
            left = _conv2d_dw(
                inputs, out_filters // 2, kernel_size, strides,
                activation=tf.nn.relu,
                use_bias=False,
                use_bn=True,
                training=training,
                name='Bypass'
            )
            right = _conv2d(
                inputs, in_filters, [1, 1],
                activation=tf.nn.relu,
                use_bias=False,
                use_bn=True,
                training=training,
                name='Convolution1x1'
            )
            right = _conv2d_dw(
                right, out_filters // 2, kernel_size, strides,
                activation=tf.nn.relu,
                use_bias=False,
                use_bn=True,
                training=training, name='DepthWiseConvolution3x3'
            )
        for i in range(1, depth):
            with tf.variable_scope('Unit{}'.format(i)):
                with tf.name_scope('ChannelShuffle'):
                    ll, lr = tf.split(left, 2, -1)
                    rl, rr = tf.split(right, 2, -1)
                    left = tf.concat([ll, rl], -1)
                    right = tf.concat([lr, rr], -1)
                right = _conv2d(
                    right, out_filters // 2, [1, 1],
                    activation=tf.nn.relu,
                    use_bias=False,
                    use_bn=True,
                    training=training,
                    name='Convolution1x1'
                )
                right = _conv2d_dw(
                    right, out_filters // 2, kernel_size, [1, 1],
                    activation=tf.nn.relu,
                    use_bias=False,
                    use_bn=True,
                    training=training,
                    name='DepthWiseConvolution3x3'
                )
        return tf.concat([left, right], -1)


def _batch_normalized_error(pred, gt_truth, num_patches=73):
    l, r = utils.norm_idx(num_patches)
    assert (l is not None and r is not None)
    norm = tf.sqrt(tf.reduce_sum((tf.square(gt_truth[:, l, :] - gt_truth[:, r, :])), 1, keepdims=True))
    return tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)) / norm


def _batch_normalized_mean_error(pred, gt_truth, num_patches=73):
    return tf.reduce_mean(_batch_normalized_error(pred, gt_truth, num_patches), 1)


def _normalized_mean_error(pred, gt_truth, num_patches=73):
    return tf.reduce_mean(_batch_normalized_mean_error(pred, gt_truth, num_patches))


class MDMModel:
    def __init__(
            self, images, shapes, mean_shape,
            batch_size, num_patches, num_channels,
            multiplier=1.0,
            is_training=True
    ):
        self.in_images = images
        self.in_shapes = shapes
        self.in_mean_shape = mean_shape
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.multiplier = multiplier
        self.is_training = is_training

        DEPTH_BASE = 232

        with tf.variable_scope('Network', values=[self.in_mean_shape], reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Initial'):
                inputs = _conv2d(
                    self.in_images, 128, [3, 3],
                    activation=tf.nn.relu,
                    use_bias=False,
                    use_bn=True,
                    training=self.is_training,
                    name='Convolution'
                )
                inputs = tf.layers.max_pooling2d(
                    inputs, [2, 2], [2, 2],
                    name='MaxPooling'
                )
            inputs = _shuffle_block(
                inputs, 128, DEPTH_BASE, [3, 3], [2, 2], 4,
                training=self.is_training,
                name='ShuffleBlock1'
            )
            inputs = _shuffle_block(
                inputs, DEPTH_BASE, 2 * DEPTH_BASE, [3, 3], [2, 2], 8,
                training=self.is_training,
                name='ShuffleBlock2'
            )
            inputs = _shuffle_block(
                inputs, 2 * DEPTH_BASE, 4 * DEPTH_BASE, [3, 3], [2, 2], 4,
                training=self.is_training,
                name='ShuffleBlock3'
            )
            with tf.variable_scope('Finalize'):
                inputs = _conv2d(
                    inputs, 1024, [1, 1],
                    activation=tf.nn.relu,
                    name='Convolution'
                )
                inputs = tf.layers.dropout(
                    inputs, 0.2,
                    training=self.is_training,
                    name='Dropout'
                )
                inputs = tf.layers.average_pooling2d(
                    inputs, [7, 7], [1, 1],
                    name='AvgPooling'
                )
            with tf.variable_scope('Predict'):
                inputs = _conv2d(
                    inputs, 146, [1, 1],
                    name='Convolution'
                )
                inputs = tf.reshape(inputs, [-1, 73, 2])
                self.prediction = inputs + self.in_mean_shape
            with tf.name_scope('BatchLoss'):
                self.batch_ne = _batch_normalized_error(self.prediction, self.in_shapes)
                self.batch_nme = tf.reduce_mean(self.batch_ne, 1)
            with tf.name_scope('Loss'):
                self.nme = tf.reduce_mean(self.batch_nme)
            tf.summary.scalar('loss', self.nme, collections=['train' if self.is_training else 'validate'])
            self.out_images, = tf.py_func(
                utils.batch_draw_landmarks_discrete,
                [self.in_images, self.in_shapes, self.prediction],
                [tf.float32]
            )
            tf.summary.image(
                'images', self.out_images,
                max_outputs=self.batch_size,
                collections=['train' if self.is_training else 'validate']
            )
