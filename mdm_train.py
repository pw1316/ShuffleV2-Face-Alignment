from functools import partial
from datetime import datetime
import data_provider
import mdm_model
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio
import detect
from menpo.shape.pointcloud import PointCloud

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('lr', 0.001, """Initial learning rate.""")
tf.flags.DEFINE_float('lr_decay_steps', 15000, """Learning rate decay steps.""")
tf.flags.DEFINE_float('lr_decay_rate', 0.1, """Learning rate decay rate.""")
tf.flags.DEFINE_integer('batch_size', 60, """The batch size to use.""")
tf.flags.DEFINE_integer('num_threads', 4, """How many pre-process threads to use.""")
tf.flags.DEFINE_string('train_dir', 'ckpt/train', """Log out directory.""")
tf.flags.DEFINE_string('pre_trained_dir', '', """Restore pre-trained model.""")
tf.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.flags.DEFINE_string(
    'datasets',
    ':'.join(
        ('Dataset/LFPW/trainset/Images/*.png',
         'Dataset/AFW/Images/*.jpg',
         'Dataset/HELEN/trainset/Images/*.jpg'
         )
    ),
    """Directory where to write event logs and checkpoint."""
)
tf.flags.DEFINE_integer('num_patches', 68, 'Landmark number')
tf.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def train(scope=''):
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/gpu:0'):
        # Global steps
        tf_global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # Learning rate
        tf_lr = tf.train.exponential_decay(
            FLAGS.lr,
            tf_global_step,
            FLAGS.lr_decay_steps,
            FLAGS.lr_decay_rate,
            staircase=True,
            name='learning_rate'
        )
        tf.summary.scalar('learning_rate', tf_lr)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(tf_lr)

        data_provider.prepare_images(FLAGS.datasets.split(':'), verbose=True)
        _mean_shape = mio.import_pickle(Path(FLAGS.train_dir) / 'reference_shape.pkl')
        with open('meta.txt', 'r') as ifs:
            _image_shape = list(map(lambda x: int(x), ifs.read().split(' ')))
        assert(isinstance(_mean_shape, np.ndarray))
        _pca_shapes = []
        _pca_bbs = []
        for item in tf.io.tf_record_iterator('pca.bin'):
            example = tf.train.Example()
            example.ParseFromString(item)
            _pca_shape = np.array(example.features.feature['pca/shape'].float_list.value).reshape((-1, 2))
            _pca_bb = np.array(example.features.feature['pca/bb'].float_list.value).reshape((-1, 2))
            _pca_shapes.append(PointCloud(_pca_shape))
            _pca_bbs.append(PointCloud(_pca_bb))
        _pca_model = detect.create_generator(_pca_shapes, _pca_bbs)
        assert(_mean_shape.shape[0] == FLAGS.num_patches)

        tf_mean_shape = tf.constant(_mean_shape, dtype=tf.float32, name='mean_shape')

        def decode_feature(serialized):
            feature = {
                'train/image': tf.FixedLenFeature([], tf.string),
                'train/shape': tf.VarLenFeature(tf.float32),
            }
            features = tf.parse_single_example(serialized, features=feature)
            decoded_image = tf.decode_raw(features['train/image'], tf.float32)
            decoded_image = tf.reshape(decoded_image, _image_shape)
            decoded_shape = tf.sparse.to_dense(features['train/shape'])
            decoded_shape = tf.reshape(decoded_shape, (FLAGS.num_patches, 2))
            return decoded_image, decoded_shape

        def get_random_sample(image, shape, rotation_stddev=10):
            # Read a random image with landmarks and bb
            image = menpo.image.Image(image.transpose((2, 0, 1)), copy=False)
            image.landmarks['PTS'] = PointCloud(shape)

            if np.random.rand() < .5:
                image = utils.mirror_image(image)
            if np.random.rand() < .5:
                theta = np.random.normal(scale=rotation_stddev)
                rot = menpo.transform.rotate_ccw_about_centre(image.landmarks['PTS'], theta)
                image = image.warp_to_shape(image.shape, rot)

            random_image = image.pixels.transpose(1, 2, 0).astype('float32')
            random_shape = image.landmarks['PTS'].points.astype('float32')
            return random_image, random_shape

        def get_random_init_shape(image, shape, mean_shape, pca):
            return image, shape, data_provider.random_shape(shape, mean_shape, pca)

        def distort_color(image, shape, init_shape):
            return data_provider.distort_color(image), shape, init_shape

        with tf.name_scope('data_provider', values=[tf_mean_shape]):
            tf_dataset = tf.data.TFRecordDataset(['train.bin'])
            tf_dataset = tf_dataset.repeat()
            tf_dataset = tf_dataset.map(decode_feature)
            tf_dataset = tf_dataset.map(
                lambda x, y: tf.py_func(
                    get_random_sample, [x, y], [tf.float32, tf.float32],
                    stateful=True,
                    name='random_sample'
                )
            )
            tf_dataset = tf_dataset.map(partial(get_random_init_shape, mean_shape=tf_mean_shape, pca=_pca_model))
            tf_dataset = tf_dataset.map(distort_color)
            tf_dataset = tf_dataset.batch(FLAGS.batch_size)
            tf_dataset = tf_dataset.prefetch(3000)
            tf_iterator = tf_dataset.make_one_shot_iterator()
            tf_images, tf_shapes, tf_initial_shapes = tf_iterator.get_next(name='batch')

        print('Defining model...')
        with tf.device(FLAGS.train_device):
            tf_model = mdm_model.MDMModel(
                tf_images,
                tf_shapes,
                tf_initial_shapes,
                num_iterations=4,
                num_patches=FLAGS.num_patches,
                patch_shape=(FLAGS.patch_size, FLAGS.patch_size)
            )
            with tf.name_scope('losses', values=tf_model.dxs + [tf_initial_shapes, tf_shapes]):
                tf_total_loss = 0
                for i, tf_dx in enumerate(tf_model.dxs):
                    with tf.name_scope('step{}'.format(i)):
                        tf_norm_error = mdm_model.normalized_rmse(
                            tf_dx + tf_initial_shapes,
                            tf_shapes,
                            num_patches=FLAGS.num_patches
                        )
                        tf_loss = tf.reduce_mean(tf_norm_error)
                    tf.summary.scalar('losses/step_{}'.format(i), tf_loss)
                    tf_total_loss += tf_loss
            tf.summary.scalar('losses/total', tf_total_loss)
            # Calculate the gradients for the batch of data
            tf_grads = opt.compute_gradients(tf_total_loss)
        tf.summary.histogram('dx', tf_model.dx)

        bn_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        # Add histograms for gradients.
        for grad, var in tf_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Apply the gradients to adjust the shared variables.
        with tf.name_scope('Optimizer', values=[tf_grads, tf_global_step]):
            apply_gradient_op = opt.apply_gradients(tf_grads, global_step=tf_global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        with tf.name_scope('MovingAverage', values=[tf_global_step]):
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, tf_global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        bn_updates_op = tf.group(*bn_updates, name='bn_group')
        train_op = tf.group(
            apply_gradient_op, variables_averages_op, bn_updates_op,
            name='train_group'
        )

        # Create a saver.
        saver = tf.train.Saver()

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        if FLAGS.pre_trained_dir:
            assert tf.gfile.Exists(FLAGS.pre_trained_dir)
            restorer = tf.train.Saver()
            restorer.restore(sess, FLAGS.pre_trained_dir)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pre_trained_dir))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        print('Starting training...')
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, tf_total_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 200 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
