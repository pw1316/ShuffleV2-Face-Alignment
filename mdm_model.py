import tensorflow as tf
import utils

extract_patches_module = tf.load_op_library('extract_patches_op/extract_patches.so')
extract_patches = extract_patches_module.extract_patches
tf.NotDifferentiable('ExtractPatches')


def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio


class MDMModel:
    def __init__(
            self, images, shapes, inits,
            batch_size=60, num_iterations=4, num_patches=68, patch_shape=(26, 26), num_channels=3,
            is_training=True
    ):
        self.in_images = images
        self.in_shapes = shapes
        self.in_init_shapes = inits
        self.num_iterations = num_iterations
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels
        self.is_training = is_training

        self.batch_size = batch_size
        self.hidden_size = 512
        self.dxs = []
        self.patches = []
        self.cnn = []
        self.rnn = []

        with tf.name_scope('Network', values=[self.in_init_shapes]):
            self.rnn_hidden = tf.zeros((self.batch_size, self.hidden_size), name='init_state')
            self.dx = tf.zeros((self.batch_size, self.num_patches, 2), name='init_dx')
            for step in range(self.num_iterations):
                with tf.name_scope('ExtractPatches{}'.format(step), values=[self.in_images, self.dx]):
                    with tf.device('/cpu:0'):
                        patches = extract_patches(
                            self.in_images,
                            tf.constant(self.patch_shape),
                            self.in_init_shapes + self.dx,
                            name='patch'
                        )
                    # self.visualize_patches(step, patches)
                    self.patches.append(patches)

                with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                    rnn_in, net = self.conv_model(patches, step)
                self.cnn.append(net)

                with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                    with tf.name_scope('mdm_rnn{}'.format(step), values=[rnn_in, self.rnn_hidden]):
                        self.rnn_hidden = tf.layers.dense(
                            tf.concat([rnn_in, self.rnn_hidden], 1), self.hidden_size,
                            activation=tf.tanh
                        )
                        prediction = tf.layers.dense(
                            self.rnn_hidden, self.num_patches * 2,
                            name='pred', activation=None
                        )
                        prediction = tf.reshape(prediction, (self.batch_size, self.num_patches, 2))
                        self.dx += prediction
                self.rnn.append(prediction)
                self.dxs.append(self.dx)
            self.prediction = tf.add(self.in_init_shapes, self.dx, name='prediction')
            self.out_images, = tf.py_func(
                utils.batch_draw_landmarks_discrete,
                [self.in_images, self.in_shapes, self.prediction],
                [tf.float32]
            )
            tf.summary.image('images', self.out_images, max_outputs=10)

    def conv_model(self, inputs, step):
        """
        Construct the CNN
        Args:
            inputs: Tensor with shape [n, num_landmarks, patch_shape, patch_shape, 3]
            step(int): RNN step
        Returns:
        """
        net = {}
        with tf.name_scope('mdm_conv{}'.format(step), values=[inputs]):
            inputs = tf.reshape(
                inputs,
                (self.batch_size * self.num_patches, self.patch_shape[0], self.patch_shape[1], self.num_channels)
            )
            # Convolution 1
            inputs = tf.layers.conv2d(inputs, 32, [3, 3], activation=None, name='conv_1')
            inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name='bn_1')
            inputs = tf.nn.relu(inputs, name='relu_1')
            self.visualize_cnn_mean(step, inputs, 'conv_1')
            net['conv_1'] = inputs
            inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2])
            net['pool_1'] = inputs

            # Convolution 2
            inputs = tf.layers.conv2d(inputs, 32, [3, 3], activation=None, name='conv_2')
            inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name='bn_2')
            inputs = tf.nn.relu(inputs, name='relu_2')
            self.visualize_cnn_mean(step, inputs, 'conv_2')
            net['conv_2'] = inputs
            inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2])
            net['pool_2'] = inputs

            # Convolution 3
            inputs = tf.layers.conv2d(inputs, 32, [3, 3], activation=None, name='conv_3')
            inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name='bn_3')
            inputs = tf.nn.relu(inputs, name='relu_3')
            self.visualize_cnn_mean(step, inputs, 'conv_3')
            net['conv_3'] = inputs
            inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2])
            net['pool_3'] = inputs
            crop_size = inputs.get_shape().as_list()[1:3]
            cropped = utils.get_central_crop(net['conv_3'], box=crop_size)
            net['conv_3_cropped'] = cropped
            inputs = tf.concat([cropped, inputs], 3)

            # Flatten
            inputs = tf.reshape(inputs, (self.batch_size, -1))
        net['concat'] = inputs
        return inputs, net

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
