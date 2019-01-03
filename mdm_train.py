from datetime import datetime
import json
import menpo
import menpo.io as mio
from menpo.shape.pointcloud import PointCloud
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import time
import utils

import data_provider
import mdm_model

tf.flags.DEFINE_string('c', 'config.json', """Model config file""")
with open(tf.flags.FLAGS.c, 'r') as g_config:
    g_config = json.load(g_config)
for k in g_config:
    print('%s:' % k, g_config[k], type(g_config[k]))
assert isinstance(g_config, dict)
input('OK?(Y/N): ')
TUNE = False


def train(scope=''):
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        # Global steps
        tf_global_step = tf.get_variable(
            'GlobalStep', [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        # Learning rate
        tf_lr = tf.train.exponential_decay(
            g_config['learning_rate'],
            tf_global_step,
            g_config['learning_rate_step'],
            g_config['learning_rate_decay'],
            staircase=True,
            name='LearningRate'
        )
        tf.summary.scalar('learning_rate', tf_lr)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(tf_lr)

        data_provider.prepare_images(
            g_config['train_dataset'].split(':'),
            num_patches=g_config['num_patches'], verbose=True
        )
        path_base = Path(g_config['train_dataset'].split(':')[0]).parent.parent
        _mean_shape = mio.import_pickle(path_base / 'reference_shape.pkl')
        _mean_shape = data_provider.align_reference_shape_to_112(_mean_shape)
        assert(isinstance(_mean_shape, np.ndarray))
        assert(_mean_shape.shape[0] == g_config['num_patches'])

        tf_mean_shape = tf.constant(_mean_shape, dtype=tf.float32, name='MeanShape')

        def decode_feature(serialized):
            feature = {
                'train/image': tf.FixedLenFeature([], tf.string),
                'train/shape': tf.VarLenFeature(tf.float32),
            }
            features = tf.parse_single_example(serialized, features=feature)
            decoded_image = tf.decode_raw(features['train/image'], tf.float32)
            decoded_image = tf.reshape(decoded_image, (336, 336, 3))
            decoded_shape = tf.sparse.to_dense(features['train/shape'])
            decoded_shape = tf.reshape(decoded_shape, (g_config['num_patches'], 2))
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
            bb = image.landmarks['PTS'].bounding_box().points
            miny, minx = np.min(bb, 0)
            maxy, maxx = np.max(bb, 0)
            bbsize = max(maxx - minx, maxy - miny)
            center = [(miny + maxy) / 2., (minx + maxx) / 2.]
            shift = (np.random.rand(2) - 0.5) / 6. * bbsize
            image.landmarks['bb'] = PointCloud(
                [
                    [center[0] - bbsize * 0.5 + shift[0], center[1] - bbsize * 0.5 + shift[1]],
                    [center[0] + bbsize * 0.5 + shift[0], center[1] + bbsize * 0.5 + shift[1]],
                ]
            ).bounding_box()
            proportion = 1.0 / 6.0 + float(np.random.rand() - 0.5) / 10.0
            image = image.crop_to_landmarks_proportion(proportion, group='bb')
            image = image.resize((112, 112))

            random_image = image.pixels.transpose(1, 2, 0).astype('float32')
            random_shape = image.landmarks['PTS'].points.astype('float32')
            return random_image, random_shape

        def distort_color(image, shape):
            return data_provider.distort_color(image), shape

        with tf.name_scope('DataProvider'):
            tf_dataset = tf.data.TFRecordDataset([str(path_base / 'train.bin')])
            tf_dataset = tf_dataset.repeat()
            tf_dataset = tf_dataset.map(decode_feature)
            tf_dataset = tf_dataset.map(
                lambda x, y: tf.py_func(
                    get_random_sample, [x, y], [tf.float32, tf.float32],
                    stateful=True,
                    name='RandomSample'
                )
            )
            tf_dataset = tf_dataset.map(distort_color)
            tf_dataset = tf_dataset.batch(g_config['batch_size'], True)
            tf_dataset = tf_dataset.prefetch(7500)
            tf_iterator = tf_dataset.make_one_shot_iterator()
            tf_images, tf_shapes = tf_iterator.get_next(name='Batch')
            tf_images.set_shape([g_config['batch_size'], 112, 112, 3])
            tf_shapes.set_shape([g_config['batch_size'], 73, 2])

        print('Defining model...')
        with tf.device(g_config['train_device']):
            tf_model = mdm_model.MDMModel(
                tf_images,
                tf_shapes,
                tf_mean_shape,
                batch_size=g_config['batch_size'],
                num_patches=g_config['num_patches'],
                num_channels=3
            )
            with tf.name_scope('Losses', values=[tf_model.prediction, tf_shapes]):
                tf_norm_error = tf_model.normalized_rmse(tf_model.prediction, tf_shapes)
                tf_loss = tf.reduce_mean(tf_norm_error)
            tf.summary.scalar('losses/total', tf_loss)
            # Calculate the gradients for the batch of data
            tf_grads = opt.compute_gradients(tf_loss)
        tf.summary.histogram('dx', tf_model.prediction - tf_shapes)

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

        with tf.name_scope('MovingAverage', values=[tf_global_step]):
            variable_averages = tf.train.ExponentialMovingAverage(g_config['MOVING_AVERAGE_DECAY'], tf_global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        bn_updates_op = tf.group(*bn_updates, name='BNGroup')
        train_op = tf.group(
            apply_gradient_op, variables_averages_op, bn_updates_op,
            name='TrainGroup'
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

        start_step = 0
        ckpt = tf.train.get_checkpoint_state(g_config['train_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /ckpt/train/model.ckpt-0,
            # extract global_step from it.
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            print('%s: Pre-trained model restored from %s' % (datetime.now(), g_config['train_dir']))
        elif TUNE:
            assign_op = []
            vvv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            cnt = 0
            for v in vvv:
                if v.name in tf_model.var_map:
                    assign_op.append(v.assign(graph.get_tensor_by_name(tf_model.var_map[v.name])))
                    cnt += 1
                else:
                    print(v.name)
            sess.run(assign_op)
            print('%s: Pre-trained model restored from graph.pb %d/%d' % (datetime.now(), cnt, len(vvv)))

        summary_writer = tf.summary.FileWriter(g_config['train_dir'], sess.graph)

        print('Starting training...')
        for step in range(start_step, g_config['max_steps']):
            start_time = time.time()
            _, loss_value = sess.run([train_op, tf_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0:
                examples_per_sec = g_config['batch_size'] / float(duration)
                format_str = (
                    '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 200 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == g_config['max_steps']:
                checkpoint_path = os.path.join(g_config['train_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
