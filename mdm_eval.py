"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import menpo
import matplotlib
import matplotlib.pyplot as plt
import mdm_model
import numpy as np
import tensorflow as tf
import time
import utils
import menpo.io as mio
from menpo.shape.pointcloud import PointCloud
import json

# Do not use a gui toolkit for matlotlib.
matplotlib.use('Agg')

tf.flags.DEFINE_string('c', 'config.json', """Model config file""")
with open(tf.flags.FLAGS.c, 'r') as g_config:
    g_config = json.load(g_config)
for k in g_config:
    print(k, type(g_config[k]), g_config[k])
input('OK?(Y/N): ')


def plot_ced(errors, method_names=['MDM']):
    from menpofit.visualize import plot_cumulative_error_distribution
    # plot the ced and store it at the root.
    fig = plt.figure()
    fig.add_subplot(111)
    plot_cumulative_error_distribution(errors, legend_entries=method_names,
                                       error_range=(0, 0.09, 0.005))
    # shift the main graph to make room for the legend
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

    
def flip_predictions(predictions, shapes):
    flipped_preds = []
    
    for pred, shape in zip(predictions, shapes):
        pred = menpo.shape.PointCloud(pred)
        if pred.points.shape[0] == 68:
            pred = utils.mirror_landmarks_68(pred, shape)
        elif pred.points.shape[0] == 73:
            pred = utils.mirror_landmarks_73(pred, shape)
        flipped_preds.append(pred.points)

    return np.array(flipped_preds, np.float32)


def evaluate():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        path_base = Path(g_config['eval_dataset']).parent.parent
        _mean_shape = mio.import_pickle(path_base / 'reference_shape.pkl')

        def decode_feature(serialized):
            feature = {
                'test/image': tf.FixedLenFeature([], tf.string),
                'test/shape': tf.VarLenFeature(tf.float32),
                'test/init': tf.VarLenFeature(tf.float32),
            }
            features = tf.parse_single_example(serialized, features=feature)
            decoded_image = tf.decode_raw(features['test/image'], tf.float32)
            decoded_image = tf.reshape(decoded_image, (256, 256, 3))
            decoded_shape = tf.sparse.to_dense(features['test/shape'])
            decoded_shape = tf.reshape(decoded_shape, (g_config['num_patches'], 2))
            decoded_init = tf.sparse.to_dense(features['test/init'])
            decoded_init = tf.reshape(decoded_init, (g_config['num_patches'], 2))
            return decoded_image, decoded_shape, decoded_init

        def get_mirrored_image(image, shape, init):
            # Read a random image with landmarks and bb
            image_m = menpo.image.Image(image.transpose((2, 0, 1)))
            image_m.landmarks['init'] = PointCloud(init)
            image_m = utils.mirror_image(image_m)
            mirrored_image = image_m.pixels.transpose(1, 2, 0).astype('float32')
            mirrored_init = image_m.landmarks['init'].points.astype('float32')
            return image, init, mirrored_image, mirrored_init, shape

        with tf.name_scope('data_provider', values=[]):
            tf_dataset = tf.data.TFRecordDataset([str(path_base / 'test.bin')])
            tf_dataset = tf_dataset.map(decode_feature)
            tf_dataset = tf_dataset.map(
                lambda x, y, z: tf.py_func(
                    get_mirrored_image, [x, y, z], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                    stateful=False,
                    name='mirror'
                )
            )
            tf_dataset = tf_dataset.batch(1)
            tf_dataset = tf_dataset.prefetch(5000)
            tf_iterator = tf_dataset.make_one_shot_iterator()
            tf_images, tf_inits, tf_images_m, tf_inits_m, tf_shapes = tf_iterator.get_next(name='batch')

        print('Loading model...')
        with tf.device(g_config['eval_device']):
            model = mdm_model.MDMModel(
                tf_images,
                tf_shapes,
                tf_inits,
                batch_size=1,
                num_iterations=g_config['num_iterations'],
                num_patches=g_config['num_patches'],
                patch_shape=(g_config['patch_size'], g_config['patch_size']),
                num_channels=3,
                is_training=False
            )
            tf.get_variable_scope().reuse_variables()
            model_m = mdm_model.MDMModel(
                tf_images_m,
                tf_shapes,
                tf_inits_m,
                batch_size=1,
                num_iterations=g_config['num_iterations'],
                num_patches=g_config['num_patches'],
                patch_shape=(g_config['patch_size'], g_config['patch_size']),
                num_channels=3,
                is_training=False
            )
        tf_predictions = model.prediction
        if g_config['use_mirror']:
            tf_predictions += tf.py_func(
                flip_predictions, (model_m.prediction, (1, 256, 256, 3)), (tf.float32, )
            )[0]
            tf_predictions /= 2.

        # Calculate predictions.
        # tf_nme = model.normalized_rmse(tf_predictions, tf_shapes)
        tf_ne = model.normalized_error(tf_predictions, tf_shapes)
        tf_nme = model.normalized_mean_error(tf_ne)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(g_config['MOVING_AVERAGE_DECAY'])
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(g_config['eval_dir'], graph_def=graph_def)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(g_config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /ckpt/train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from {} at step={}.'.format(ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            eval_base = Path('Evaluate')
            for i in range(10):
                eval_path = eval_base / 'err{}'.format(i)
                if not eval_path.exists():
                    eval_path.mkdir(parents=True)

            num_iter = g_config['num_examples']
            # Counts the number of correct predictions.
            errors = []
            mean_errors = []

            print('%s: starting evaluation on (%s).' % (datetime.now(), g_config['eval_dataset']))
            start_time = time.time()
            for step in range(num_iter):
                rmse, rse, img = sess.run([tf_nme, tf_ne, model.out_images])
                error_level = min(9, int(rmse[0] * 100))
                plt.imsave('Evaluate/err{}/step{}.png'.format(error_level, step), img[0])
                errors.append(rse)
                mean_errors.append(rmse)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = 1. / sec_per_batch
                    log_str = '{}: [{:d} batches out of {:d}] ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(log_str.format(datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()

            errors = np.array(errors)
            errors = np.reshape(errors, (-1, g_config['num_patches']))
            print(errors.shape)
            mean_errors = np.vstack(mean_errors).ravel()
            mean_rse = np.mean(errors, 0)
            mean_rmse = mean_errors.mean()
            with open('Evaluate/errors.txt', 'w') as ofs:
                for row, avg in zip(errors, mean_errors):
                    for col in row:
                        ofs.write('%.4f, ' % col)
                    ofs.write('%.4f' % avg)
                    ofs.write('\n')
                for col in mean_rse:
                    ofs.write('%.4f, ' % col)
                ofs.write('%.4f' % mean_rmse)
                ofs.write('\n')
            auc_at_08 = (mean_errors < .08).mean()
            auc_at_05 = (mean_errors < .05).mean()
            ced_image = plot_ced([mean_errors.tolist()])
            ced_plot = sess.run(tf.summary.merge([tf.summary.image('ced_plot', ced_image[None, ...])]))

            print('Errors', mean_errors.shape)
            print(
                '%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f [%d examples]' %
                (datetime.now(), mean_errors.mean(), auc_at_05, auc_at_08, num_iter)
            )
            summary_writer.add_summary(ced_plot, global_step)


if __name__ == '__main__':
    evaluate()
