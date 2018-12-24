import tensorflow as tf
import utils

# extract_patches_module = tf.load_op_library('extract_patches_op/extract_patches.so')
# extract_patches = extract_patches_module.extract_patches
# tf.NotDifferentiable('ExtractPatches')


def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio


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
        inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding='same', use_bias=use_bias)
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training)
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
        name='ConvolutionDepthWise'
):
    with tf.variable_scope(name, values=[inputs]):
        layer = tf.keras.layers.DepthwiseConv2D(
            kernel_size, strides,
            padding='same', use_bias=use_bias)
        inputs = layer.apply(inputs)
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training)
        inputs = tf.layers.conv2d(inputs, filters, [1, 1], padding='same', use_bias=use_bias)
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training)
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
            left = _conv2d(
                inputs, in_filters, [1, 1],
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution1x1'
            )
            left = _conv2d_dw(
                left, out_filters // 2, kernel_size, strides,
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution3x3DepthWise'
            )
            right = _conv2d_dw(
                inputs, out_filters // 2, kernel_size, strides,
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Bypass'
            )
        for i in range(1, depth):
            with tf.variable_scope('Unit{}'.format(i)):
                with tf.name_scope('ChannelShuffle'):
                    ll, lr = tf.split(left, [out_filters // 4, out_filters // 4], -1)
                    rl, rr = tf.split(right, [out_filters // 4, out_filters // 4], -1)
                    left = tf.concat([ll, rr], -1)
                    right = tf.concat([rl, lr], -1)
                left = _conv2d(
                    left, out_filters // 2, [1, 1],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution1x1'
                )
                left = _conv2d_dw(
                    left, out_filters // 2, kernel_size, [1, 1],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution3x3DepthWise'
                )
        return tf.concat([left, right], -1)


class MDMModel:
    def __init__(
            self, images, shapes, inits,
            batch_size, num_iterations, num_patches, patch_shape, num_channels,
            is_training=True
    ):
        self.in_images = images
        self.in_shapes = shapes
        self.in_init_shapes = inits
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels
        self.is_training = is_training

        with tf.variable_scope('Network', values=[self.in_init_shapes]):
            with tf.variable_scope('Initial'):
                inputs = _conv2d(
                    self.in_images, 64, [3, 3],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=self.is_training, name='Convolution'
                )
                inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], name='MaxPooling')
            inputs = _shuffle_block(
                inputs, 64, 96, [3, 3], [2, 2], 4,
                training=self.is_training, name='ShuffleBlock1'
            )
            inputs = _shuffle_block(
                inputs, 96, 192, [3, 3], [2, 2], 8,
                training=self.is_training, name='ShuffleBlock2'
            )
            inputs = _shuffle_block(
                inputs, 192, 384, [3, 3], [2, 2], 4,
                training=self.is_training, name='ShuffleBlock3'
            )
            with tf.variable_scope('Finalize'):
                inputs = _conv2d(inputs, 1024, [1, 1], activation=tf.nn.relu, name='Convolution')
                inputs = tf.layers.dropout(inputs, 0.8, training=self.is_training, name='Dropout')
                inputs = tf.layers.average_pooling2d(inputs, [24, 24], [1, 1], name='AvgPooling')
            with tf.variable_scope('Predict'):
                inputs = _conv2d(inputs, 146, [1, 1], name='Convolution')
                inputs = tf.reshape(inputs, [-1, 73, 2])
                self.prediction = inputs + self.in_init_shapes
            self.out_images, = tf.py_func(
                utils.batch_draw_landmarks_discrete,
                [self.in_images, self.in_shapes, self.prediction],
                [tf.float32]
            )
            tf.summary.image('images', self.out_images, max_outputs=10)

    def visualize_patches(self, step, inputs):
        """
        Visualize Feature Map
        Args:
            step(int): RNN step
            inputs: Tensor with shape [n, num_landmarks, patch_shape, patch_shape, 3]
        Returns:
            None
        """
        with tf.name_scope('visualize', values=[inputs]):
            inputs = inputs[:10]
            inputs = tf.transpose(inputs, (0, 2, 1, 3, 4))
            inputs = tf.reshape(inputs, (1, -1, self.num_patches * self.patch_shape[1], 3))
        tf.summary.image('patches/step{}'.format(step), inputs)

    def visualize_cnn(self, step, inputs, name):
        """
        Visualize Feature Map
        Args:
            step(int): RNN step
            inputs: Tensor with shape [n * num_landmarks, h, w, c]
            name(str): Image name
        Returns:
            None
        """
        tf.summary.image(
            'feature_step{}_{}'.format(step, name),
            tf.reshape(
                tf.transpose(inputs[:10 * self.num_patches], perm=[0, 1, 3, 2]),
                (min(10, self.batch_size), self.num_patches * inputs.shape[1], inputs.shape[2] * inputs.shape[3], 1)
            ),
            max_outputs=min(10, self.batch_size)
        )

    def visualize_cnn_mean(self, step, inputs, name):
        """
        Visualize Mean Feature Map
        Args:
            step(int): RNN step
            inputs: Tensor with shape [n * num_landmarks, h, w, c]
            name(str): Image name
        Returns:
            None
        """
        with tf.name_scope('visualize_{}'.format(name), values=[inputs]):
            inputs = tf.reduce_mean(inputs, 3)
            inputs = tf.reshape(inputs, (self.batch_size, self.num_patches, inputs.shape[1], inputs.shape[2]))
            inputs = inputs[:10]
            inputs = tf.transpose(inputs, (0, 2, 1, 3))
            inputs = tf.reshape(inputs, (1, inputs.shape[0] * inputs.shape[1], -1, 1))
        tf.summary.image('cnn_mean_feature/step{}/{}'.format(step, name), inputs)

    def normalized_rmse(self, pred, gt_truth):
        l, r = utils.norm_idx(self.num_patches)
        assert (l is not None and r is not None)
        norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, l, :] - gt_truth[:, r, :]) ** 2), 1))
        return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * self.num_patches)

    def normalized_error(self, pred, gt_truth):
        l, r = utils.norm_idx(self.num_patches)
        assert (l is not None and r is not None)
        norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, l, :] - gt_truth[:, r, :]) ** 2), 1))
        return tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)) / norm

    def normalized_mean_error(self, n_error):
        return tf.reduce_sum(n_error, 1) / self.num_patches
