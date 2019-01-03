import tensorflow as tf
import utils


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
        inputs = tf.layers.conv2d(
            inputs, filters, kernel_size, strides, padding='same', use_bias=use_bias, name='Conv2D'
        )
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training, name='BatchNorm')
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
        layer = tf.keras.layers.DepthwiseConv2D(
            kernel_size, strides,
            padding='same', use_bias=use_bias, name='DWConv2D')
        inputs = layer.apply(inputs)
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training, name='BatchNorm1')
        inputs = tf.layers.conv2d(inputs, filters, [1, 1], padding='same', use_bias=use_bias, name='Conv2D')
        if use_bn:
            inputs = tf.layers.batch_normalization(inputs, training=training, name='BatchNorm2')
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
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Bypass'
            )
            right = _conv2d(
                inputs, in_filters, [1, 1],
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution1x1'
            )
            right = _conv2d_dw(
                right, out_filters // 2, kernel_size, strides,
                activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='DepthWiseConvolution3x3'
            )
        for i in range(1, depth):
            with tf.variable_scope('Unit{}'.format(i)):
                with tf.name_scope('ChannelShuffle'):
                    ll, lr = tf.split(left, [out_filters // 4, out_filters // 4], -1)
                    rl, rr = tf.split(right, [out_filters // 4, out_filters // 4], -1)
                    left = tf.concat([ll, rl], -1)
                    right = tf.concat([lr, rr], -1)
                right = _conv2d(
                    right, out_filters // 2, [1, 1],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='Convolution1x1'
                )
                right = _conv2d_dw(
                    right, out_filters // 2, kernel_size, [1, 1],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=training, name='DepthWiseConvolution3x3'
                )
        return tf.concat([left, right], -1)


class MDMModel:
    def __init__(
            self, images, shapes, mean_shape,
            batch_size, num_patches, num_channels,
            is_training=True
    ):
        self.in_images = images
        self.in_shapes = shapes
        self.in_mean_shape = mean_shape
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.is_training = is_training

        with tf.variable_scope('Network', values=[self.in_mean_shape]):
            with tf.variable_scope('Initial'):
                inputs = _conv2d(
                    self.in_images, 128, [3, 3],
                    activation=tf.nn.relu, use_bias=False, use_bn=True, training=self.is_training, name='Convolution'
                )
                inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2], name='MaxPooling')
            inputs = _shuffle_block(
                inputs, 128, 232, [3, 3], [2, 2], 4,
                training=self.is_training, name='ShuffleBlock1'
            )
            inputs = _shuffle_block(
                inputs, 232, 464, [3, 3], [2, 2], 8,
                training=self.is_training, name='ShuffleBlock2'
            )
            inputs = _shuffle_block(
                inputs, 464, 928, [3, 3], [2, 2], 4,
                training=self.is_training, name='ShuffleBlock3'
            )
            with tf.variable_scope('Finalize'):
                inputs = _conv2d(inputs, 1024, [1, 1], activation=tf.nn.relu, name='Convolution')
                inputs = tf.layers.dropout(inputs, 0.2, training=self.is_training, name='Dropout')
                inputs = tf.layers.average_pooling2d(inputs, [7, 7], [1, 1], name='AvgPooling')
            with tf.variable_scope('Predict'):
                inputs = _conv2d(inputs, 146, [1, 1], name='Convolution')
                inputs = tf.reshape(inputs, [-1, 73, 2])
                self.prediction = inputs + self.in_mean_shape
            self.out_images, = tf.py_func(
                utils.batch_draw_landmarks_discrete,
                [self.in_images, self.in_shapes, self.prediction],
                [tf.float32]
            )
            tf.summary.image('images', self.out_images, max_outputs=10)

        # For tuning
        with tf.gfile.GFile('graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='Original')
        self.var_map = {}
        for i in range(4):
            self.map_shuffle_block(1, i)
        for i in range(8):
            self.map_shuffle_block(2, i)
        for i in range(4):
            self.map_shuffle_block(3, i)
        self.var_map['Network/Finalize/Convolution/Conv2D/kernel:0'] = 'Original/Stage1/Conv5/weights:0'
        self.var_map['Network/Finalize/Convolution/Conv2D/bias:0'] = 'Original/Stage1/Conv5/biases:0'

    def map_conv2d_bn(self, src, dst):
        self.var_map[dst + '/Conv2D/kernel:0'] = src + '/weights:0'
        self.var_map[dst + '/BatchNorm/beta:0'] = src + '/BatchNorm/beta:0'
        self.var_map[dst + '/BatchNorm/gamma:0'] = src + '/BatchNorm/Const:0'
        self.var_map[dst + '/BatchNorm/moving_mean:0'] = src + '/BatchNorm/moving_mean:0'
        self.var_map[dst + '/BatchNorm/moving_variance:0'] = src + '/BatchNorm/moving_variance:0'

    def map_conv2d_dw_bn(self, src, dst):
        self.var_map[dst + '/DWConv2D/depthwise_kernel:0'] = src + '/SeparableConv2d/depthwise_weights:0'
        self.var_map[dst + '/BatchNorm1/beta:0'] = src + '/SeparableConv2d/BatchNorm/beta:0'
        self.var_map[dst + '/BatchNorm1/gamma:0'] = src + '/SeparableConv2d/BatchNorm/Const:0'
        self.var_map[dst + '/BatchNorm1/moving_mean:0'] = src + '/SeparableConv2d/BatchNorm/moving_mean:0'
        self.var_map[dst + '/BatchNorm1/moving_variance:0'] = src + '/SeparableConv2d/BatchNorm/moving_variance:0'
        self.var_map[dst + '/Conv2D/kernel:0'] = src + '/conv1x1_after/weights:0'
        self.var_map[dst + '/BatchNorm2/beta:0'] = src + '/conv1x1_after/BatchNorm/beta:0'
        self.var_map[dst + '/BatchNorm2/gamma:0'] = src + '/conv1x1_after/BatchNorm/Const:0'
        self.var_map[dst + '/BatchNorm2/moving_mean:0'] = src + '/conv1x1_after/BatchNorm/moving_mean:0'
        self.var_map[dst + '/BatchNorm2/moving_variance:0'] = src + '/conv1x1_after/BatchNorm/moving_variance:0'

    def map_shuffle_block(self, sid, uid):
        self.map_conv2d_bn(
            'Original/Stage1/Stage{}/unit_{}/conv1x1_before'.format(sid + 1, uid + 1),
            'Network/ShuffleBlock{}/Unit{}/Convolution1x1'.format(sid, uid)
        )
        self.map_conv2d_dw_bn(
            'Original/Stage1/Stage{}/unit_{}'.format(sid + 1, uid + 1),
            'Network/ShuffleBlock{}/Unit{}/DepthWiseConvolution3x3'.format(sid, uid)
        )
        if uid == 0:
            self.map_conv2d_dw_bn(
                'Original/Stage1/Stage{}/unit_{}/second_branch'.format(sid + 1, uid + 1),
                'Network/ShuffleBlock{}/Unit{}/Bypass'.format(sid, uid)
            )

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
