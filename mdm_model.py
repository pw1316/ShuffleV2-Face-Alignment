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


def normalized_rmse(pred, gt_truth, num_patches=68):
    l, r = utils.norm_idx(num_patches)
    assert(l is not None and r is not None)
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, l, :] - gt_truth[:, r, :])**2), 1))
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * num_patches)


class MDMModel:
    def __init__(self, images, inits, num_iterations=4, num_patches=68, patch_shape=(26, 26), num_channels=3):
        self.in_images = images
        self.in_init_shapes = inits
        self.num_iterations = num_iterations
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels

        self.batch_size = images.get_shape().as_list()[0]
        self.rnn_hidden = tf.zeros((self.batch_size, 512))
        self.dx = tf.zeros((self.batch_size, self.num_patches, 2))
        self.dxs = []
        self.patches = []
        self.cnn = []
        self.rnn = []

        for step in range(self.num_iterations):
            with tf.device('/cpu:0'):
                patches = extract_patches(self.in_images, tf.constant(self.patch_shape), self.in_init_shapes + self.dx)
            self.visualize_patches(step, patches)
            self.patches.append(patches)

            with tf.variable_scope('convnet', reuse=step > 0):
                rnn_in, net = self.conv_model(patches, step)
                self.cnn.append(net)

            with tf.variable_scope('rnn', reuse=step > 0):
                self.rnn_hidden = tf.layers.dense(tf.concat([rnn_in, self.rnn_hidden], 1), 512, activation=tf.tanh)
                prediction = tf.layers.dense(self.rnn_hidden, self.num_patches * 2, name='pred', activation=None)
                prediction = tf.reshape(prediction, (self.batch_size, self.num_patches, 2))
                self.rnn.append(prediction)
            self.dx += prediction
            self.dxs.append(self.dx)
        self.pred = self.in_init_shapes + self.dx

    def conv_model(self, inputs, step, is_training=True, scope=''):
        """
        Construct the CNN
        Args:
            inputs: Tensor with shape [n, num_landmarks, patch_shape, patch_shape, 3]
            step(int): RNN step
            is_training(bool): Is training or not
            scope(str): Scope
        Returns:
        """
        inputs = tf.reshape(
            inputs,
            (self.batch_size * self.num_patches, self.patch_shape[0], self.patch_shape[1], self.num_channels)
        )
        net = {}
        with tf.name_scope(scope, 'mdm_conv', [inputs]):
            inputs = tf.layers.conv2d(inputs, 32, [3, 3], activation=tf.nn.relu, name='conv_1')
            self.visualize_cnn_mean(step, inputs, 'conv_1')
            net['conv_1'] = inputs

            inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2])
            self.visualize_cnn_mean(step, inputs, 'pool_1')
            net['pool_1'] = inputs

            inputs = tf.layers.conv2d(inputs, 32, [3, 3], activation=tf.nn.relu, name='conv_2')
            self.visualize_cnn_mean(step, inputs, 'conv_2')
            net['conv_2'] = inputs

            inputs = tf.layers.max_pooling2d(inputs, [2, 2], [2, 2])
            self.visualize_cnn_mean(step, inputs, 'pool_2')
            net['pool_2'] = inputs

            crop_size = inputs.get_shape().as_list()[1:3]
            cropped = utils.get_central_crop(net['conv_2'], box=crop_size)
            self.visualize_cnn_mean(step, cropped, 'conv_2_cropped')
            net['conv_2_cropped'] = cropped

            inputs = tf.reshape(tf.concat([cropped, inputs], 3), (self.batch_size, -1))
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
        inputs = tf.reduce_mean(inputs, 3)
        inputs = tf.reshape(inputs, (self.batch_size, self.num_patches, inputs.shape[1], inputs.shape[2]))
        inputs = inputs[:10]
        inputs = tf.transpose(inputs, (0, 2, 1, 3))
        inputs = tf.reshape(inputs, (1, inputs.shape[0] * inputs.shape[1], -1, 1))
        tf.summary.image('cnn_mean_feature/step{}/{}'.format(step, name), inputs)
