from datetime import datetime
import data_provider
import mdm_model
import numpy as np
import os
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.001, """Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay_steps', 15000, """Learning rate decay steps.""")
tf.app.flags.DEFINE_float('lr_decay_rate', 0.1, """Learning rate decay rate.""")
tf.app.flags.DEFINE_integer('batch_size', 60, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_threads', 4, """How many pre-process threads to use.""")
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train', """Log out directory.""")
tf.app.flags.DEFINE_string('pre_trained_dir', '', """Restore pre-trained model.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string(
    'datasets',
    ':'.join(
        ('Dataset/LFPW/trainset/Images/*.png',
         'Dataset/AFW/Images/*.jpg',
         'Dataset/HELEN/trainset/Images/*.jpg'
         )
    ),
    """Directory where to write event logs and checkpoint."""
)
tf.app.flags.DEFINE_integer('num_patches', 68, 'Landmark number')
tf.app.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')

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

        train_dirs = FLAGS.datasets.split(':')
        _images, _shapes, _mean_shape, _pca_model = \
            data_provider.load_images(train_dirs, verbose=True)
        assert(_shapes[0].points.shape[0] == FLAGS.num_patches)
        assert(_mean_shape.shape[0] == FLAGS.num_patches)

        tf_mean_shape = tf.constant(_mean_shape, dtype=tf.float32, name='mean_shape')

        def get_random_sample(rotation_stddev=10):
            random_idx = np.random.randint(low=0, high=len(_images))
            im = menpo.image.Image(_images[random_idx].transpose(2, 0, 1), copy=False)
            random_shape = _shapes[random_idx]
            im.landmarks['PTS'] = random_shape
            if np.random.rand() < .5:
                im = utils.mirror_image(im)
            if np.random.rand() < .5:
                theta = np.random.normal(scale=rotation_stddev)
                rot = menpo.transform.rotate_ccw_about_centre(random_shape, theta)
                im = im.warp_to_shape(im.shape, rot)

            random_image = im.pixels.transpose(1, 2, 0).astype('float32')
            random_shape = im.landmarks['PTS'].points.astype('float32')
            return random_image, random_shape

        tf_image, tf_shape = tf.py_func(get_random_sample, [], [tf.float32, tf.float32], stateful=True)
        tf_initial_shape = data_provider.random_shape(tf_shape, tf_mean_shape, _pca_model)
        tf_image.set_shape(_images[0].shape)
        tf_shape.set_shape(_shapes[0].points.shape)
        tf_initial_shape.set_shape(_shapes[0].points.shape)
        tf_image = data_provider.distort_color(tf_image)

        tf_images, tf_shapes, tf_initial_shapes = tf.train.batch(
            [tf_image, tf_shape, tf_initial_shape],
            FLAGS.batch_size,
            dynamic_pad=False,
            capacity=5000,
            enqueue_many=False,
            num_threads=FLAGS.num_threads,
            name='batch'
        )

        print('Defining model...')
        with tf.device(FLAGS.train_device):
            tf_model = mdm_model.MDMModel(
                tf_images,
                tf_initial_shapes,
                num_iterations=4,
                num_patches=FLAGS.num_patches,
                patch_shape=(FLAGS.patch_size, FLAGS.patch_size)
            )
            tf_total_loss = 0
            for i, tf_dx in enumerate(tf_model.dxs):
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

        pred_images, = tf.py_func(utils.batch_draw_landmarks, [tf_images, tf_model.pred], [tf.float32])
        gt_images, = tf.py_func(utils.batch_draw_landmarks, [tf_images, tf_shapes], [tf.float32])
        tf.summary.image('images', tf.concat([gt_images, pred_images], 2), max_outputs=5)
        tf.summary.histogram('dx', tf_model.dx)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION, scope)

        # Add histograms for gradients.
        for grad, var in tf_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(tf_grads, global_step=tf_global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, tf_global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        if FLAGS.pre_trained_dir:
            assert tf.gfile.Exists(FLAGS.pre_trained_dir)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pre_trained_dir)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pre_trained_dir))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

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
