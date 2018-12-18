"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import utils
import math
from menpo.shape.pointcloud import PointCloud
import menpo.io as mio

extract_patches_module = tf.load_op_library('extract_patches_op/extract_patches.so')
extract_patches = extract_patches_module.extract_patches
tf.NotDifferentiable('ExtractPatches')
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('inf_dir', 'ckpt/inf', """Directory where to write event logs.""")
tf.flags.DEFINE_string('dataset', 'Dataset/300W/*/Images/*.png', """The dataset path to evaluate.""")
MDM_MODEL_PATH = 'theano_mdm.pb'


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


def normalized_error(pred, gt_truth):
    norm = np.sqrt(np.sum(np.square(gt_truth[36, :] - gt_truth[45, :])))
    return np.sqrt(np.sum(np.square(pred - gt_truth), 1)) / norm


def normalized_mean_error(n_error):
    assert np.sum(n_error) / 68 == np.mean(n_error)
    return np.mean(n_error)


def influence():
    image_paths = sorted(list(Path('.').glob(FLAGS.dataset)))
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        tf_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='images')
        tf_init = tf.placeholder(tf.float32, shape=4, name='inits')
        with open(MDM_MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())
            tf_prediction, = tf.import_graph_def(
                graph_def,
                input_map={"image": tf_image, "bounding_box": tf_init},
                return_elements=['prediction:0']
            )

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            errors = []
            mean_errors = []

            step = 0
            start_time = time.time()
            for path in image_paths:
                mp_image = mio.import_image(path)
                if mp_image.n_channels == 1:
                    mp_image.pixels = np.vstack([mp_image.pixels] * 3)
                mp_image.landmarks['bb'] = mio.import_landmark_file(
                    str(Path(path.parent.parent / 'BoundingBoxes' / (path.stem + '.pts')))
                )
                ly, lx = mp_image.landmarks['bb'].points[0]
                hy, hx = mp_image.landmarks['bb'].points[2]
                cx = (lx + hx) / 2
                cy = (ly + hy) / 2
                bb_size = int(math.ceil(max(hx - lx, hy - ly)))
                square_bb = np.array([[cy - bb_size / 2, cx - bb_size / 2], [cy + bb_size / 2, cx + bb_size / 2]])
                mp_image.landmarks['square_bb'] = PointCloud(square_bb)
                mp_image = mp_image.crop_to_landmarks_proportion(0.3, group='square_bb')
                mp_image = mp_image.rescale_to_diagonal(320)

                np_image = mp_image.pixels.transpose((1, 2, 0))
                np_shape = mp_image.landmarks['PTS'].points
                np_init = mp_image.landmarks['bb'].points

                prediction, = sess.run(tf_prediction, feed_dict={
                    tf_image: np_image,
                    # grab the upper-left and lower-down points of the bounding box.
                    tf_init: np_init[[0, 2]].ravel()
                })
                error = normalized_error(prediction, mp_image.landmarks['PTS'].points)
                mean_error = normalized_mean_error(error)
                error_level = min(9, int(mean_error * 100))

                concat_image = utils.draw_landmarks_discrete(
                    np_image,
                    np_shape,
                    prediction
                )
                plt.imsave('err{}/step{}.png'.format(error_level, step), concat_image)
                errors.append(error)
                mean_errors.append(mean_error)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = 1. / sec_per_batch
                    log_str = '{}: [{:d} batches done] ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(log_str.format(datetime.now(), step, examples_per_sec, sec_per_batch))
                    start_time = time.time()

            errors = np.array(errors)
            print(errors.shape)
            mean_errors = np.vstack(mean_errors).ravel()
            errors_mean = np.mean(errors, 0)
            mean_errors_mean = mean_errors.mean()
            with open('errors.txt', 'w') as ofs:
                for row, avg in zip(errors, mean_errors):
                    for col in row:
                        ofs.write('%.4f, ' % col)
                    ofs.write('%.4f' % avg)
                    ofs.write('\n')
                for col in errors_mean:
                    ofs.write('%.4f, ' % col)
                ofs.write('%.4f' % mean_errors_mean)
                ofs.write('\n')
            auc_at_08 = (mean_errors < .08).mean()
            auc_at_05 = (mean_errors < .05).mean()
            # ced_image = plot_ced([mean_errors.tolist()])
            # ced_plot = ced_image[None, ...]
            # plt.imshow(ced_plot)
            # plt.show()

            print('Errors', mean_errors.shape)
            print(
                '%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f' %
                (datetime.now(), mean_errors.mean(), auc_at_05, auc_at_08)
            )


if __name__ == '__main__':
    influence()
