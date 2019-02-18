from datetime import datetime
import menpo
import menpo.io as mio
from menpo.shape.pointcloud import PointCloud
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import time

import data_provider
import mdm_model
import utils

g_config = utils.load_config()


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
        tf.summary.scalar('learning_rate', tf_lr, collections=['train'])

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
        _negatives = []
        for mp_image in mio.import_images('Dataset/Neg/*.png', verbose=True):
            _negatives.append(mp_image.pixels.transpose(1, 2, 0).astype(np.float32))
        _num_negatives = len(_negatives)

        tf_mean_shape = tf.constant(_mean_shape, dtype=tf.float32, name='MeanShape')

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
            # shift = np.zeros(2, np.float32)
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

            # Occlude
            _O_AREA = 0.15
            _O_MIN_H = 0.15
            _O_MAX_H = 1.0
            if np.random.rand() < .3:
                rh = min(112, int((np.random.rand() * (_O_MAX_H - _O_MIN_H) + _O_MIN_H) * 112))
                rw = min(112, int(12544 * _O_AREA / rh))
                dy = int(np.random.rand() * (112 - rh))
                dx = int(np.random.rand() * (112 - rw))
                idx = int(np.random.rand() * _num_negatives)
                random_image[dy:dy+rh, dx:dx+rw] = np.minimum(
                    1.0,
                    _negatives[idx][dy:dy+rh, dx:dx+rw]
                )

            return random_image, random_shape

        def decode_feature_and_augment(serialized):
            feature = {
                'train/image': tf.FixedLenFeature([], tf.string),
                'train/shape': tf.VarLenFeature(tf.float32),
            }
            features = tf.parse_single_example(serialized, features=feature)
            decoded_image = tf.decode_raw(features['train/image'], tf.float32)
            decoded_image = tf.reshape(decoded_image, (336, 336, 3))
            decoded_shape = tf.sparse.to_dense(features['train/shape'])
            decoded_shape = tf.reshape(decoded_shape, (g_config['num_patches'], 2))

            random_image, random_shape = tf.py_func(
                get_random_sample, [decoded_image, decoded_shape], [tf.float32, tf.float32],
                stateful=True,
                name='RandomSample'
            )
            return data_provider.distort_color(random_image), random_shape

        def decode_feature(serialized):
            feature = {
                'validate/image': tf.FixedLenFeature([], tf.string),
                'validate/shape': tf.VarLenFeature(tf.float32),
            }
            features = tf.parse_single_example(serialized, features=feature)
            decoded_image = tf.decode_raw(features['validate/image'], tf.float32)
            decoded_image = tf.reshape(decoded_image, (112, 112, 3))
            decoded_shape = tf.sparse.to_dense(features['validate/shape'])
            decoded_shape = tf.reshape(decoded_shape, (g_config['num_patches'], 2))
            return decoded_image, decoded_shape

        with tf.name_scope('DataProvider'):
            tf_dataset = tf.data.TFRecordDataset([str(path_base / 'train.bin')])
            tf_dataset = tf_dataset.repeat()
            tf_dataset = tf_dataset.map(decode_feature_and_augment, num_parallel_calls=5)
            tf_dataset = tf_dataset.batch(g_config['batch_size'], True)
            tf_dataset = tf_dataset.prefetch(1)
            tf_iterator = tf_dataset.make_one_shot_iterator()
            tf_images, tf_shapes = tf_iterator.get_next(name='Batch')
            tf_images.set_shape([g_config['batch_size'], 112, 112, 3])
            tf_shapes.set_shape([g_config['batch_size'], 73, 2])

            tf_dataset_v = tf.data.TFRecordDataset([str(path_base / 'validate.bin')])
            tf_dataset_v = tf_dataset_v.repeat()
            tf_dataset_v = tf_dataset_v.map(decode_feature, num_parallel_calls=5)
            tf_dataset_v = tf_dataset_v.batch(50, True)
            tf_dataset_v = tf_dataset_v.prefetch(1)
            tf_iterator_v = tf_dataset_v.make_one_shot_iterator()
            tf_images_v, tf_shapes_v = tf_iterator_v.get_next(name='ValidateBatch')
            tf_images_v.set_shape([50, 112, 112, 3])
            tf_shapes_v.set_shape([50, 73, 2])

        print('Defining model...')
        with tf.device(g_config['train_device']):
            tf_model = mdm_model.MDMModel(
                tf_images,
                tf_shapes,
                tf_mean_shape,
                batch_size=g_config['batch_size'],
                num_patches=g_config['num_patches'],
                num_channels=3,
                multiplier=g_config['multiplier']
            )
            tf_grads = opt.compute_gradients(tf_model.nme)
            with tf.name_scope('Validate'):
                tf_model_v = mdm_model.MDMModel(
                    tf_images_v,
                    tf_shapes_v,
                    tf_mean_shape,
                    batch_size=50,
                    num_patches=g_config['num_patches'],
                    num_channels=3,
                    multiplier=g_config['multiplier'],
                    is_training=False
                )
        tf.summary.histogram('dx', tf_model.prediction - tf_shapes, collections=['train'])

        bn_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        # Add histograms for gradients.
        for grad, var in tf_grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad, collections=['train'])

        # Apply the gradients to adjust the shared variables.
        with tf.name_scope('Optimizer', values=[tf_grads, tf_global_step]):
            apply_gradient_op = opt.apply_gradients(tf_grads, global_step=tf_global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

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

        train_summary_op = tf.summary.merge_all('train')
        validate_summary_op = tf.summary.merge_all('validate')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        init = tf.global_variables_initializer()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        # Assuming model_checkpoint_path looks something like:
        #   /ckpt/train/model.ckpt-0,
        # extract global_step from it.
        start_step = 0
        ckpt = tf.train.get_checkpoint_state(g_config['train_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            print('%s: Restart from %s' % (datetime.now(), g_config['train_dir']))
        else:
            ckpt = tf.train.get_checkpoint_state(g_config['ckpt_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                tf_global_step_op = tf_global_step.assign(0)
                sess.run(tf_global_step_op)
                print('%s: Pre-trained model restored from %s' % (datetime.now(), g_config['ckpt_dir']))

        train_writer = tf.summary.FileWriter(g_config['train_dir'] + '/train', sess.graph)
        validate_writer = tf.summary.FileWriter(g_config['train_dir'] + '/validate', sess.graph)

        print('Starting training...')
        steps_per_epoch = 15000 / g_config['batch_size']
        for step in range(start_step, g_config['max_steps']):
            if step % steps_per_epoch == 0:
                start_time = time.time()
                _, train_loss, train_summary = sess.run([train_op, tf_model.nme, train_summary_op])
                duration = time.time() - start_time
                validate_loss, validate_summary = sess.run([tf_model_v.nme, validate_summary_op])
                train_writer.add_summary(train_summary, step)
                validate_writer.add_summary(validate_summary, step)

                print(
                    '%s: step %d, loss = %.4f (%.3f sec/batch)' % (
                        datetime.now(), step, train_loss, duration
                    )
                )
                print(
                    '%s: step %d, validate loss = %.4f' % (
                        datetime.now(), step, validate_loss
                    )
                )
            else:
                start_time = time.time()
                _, train_loss = sess.run([train_op, tf_model.nme])
                duration = time.time() - start_time
                if step % 100 == 0:
                    print(
                        '%s: step %d, loss = %.4f (%.3f sec/batch)' % (
                            datetime.now(), step, train_loss, duration
                        )
                    )

            assert not np.isnan(train_loss), 'Model diverged with loss = NaN'

            if step % steps_per_epoch == 0 or (step + 1) == g_config['max_steps']:
                checkpoint_path = os.path.join(g_config['train_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
