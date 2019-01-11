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
import menpo.image
import menpo.io as mio
import cv2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('inf_dir', 'ckpt/inf', """Directory where to write event logs.""")
tf.flags.DEFINE_string('dataset', 'Dataset/300W/*/Images/*.png', """The dataset path to evaluate.""")
MDM_MODEL_PATH = 'graph.pb'


def normalized_batch_nme(pred, gt_truth):
    norm = np.sqrt(np.sum(np.square(gt_truth[36, :] - gt_truth[45, :])))
    return np.sqrt(np.sum(np.square(pred - gt_truth), 1)) / norm


def normalized_nme(n_error):
    assert np.sum(n_error) / 68 == np.mean(n_error)
    return np.mean(n_error)


def influence():
    image_paths = sorted(list(Path('.').glob(FLAGS.dataset)))
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with open(MDM_MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef.FromString(f.read())
            tf.import_graph_def(graph_def)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            errors = []
            mean_errors = []

            step = 0
            start_time = time.time()
            for path in image_paths:
                mp_image = mio.import_image(path)
                assert isinstance(mp_image, menpo.image.Image)
                if mp_image.n_channels == 3:
                    mp_image.pixels = np.mean(mp_image.pixels, 0, keepdims=True)
                mp_image.landmarks['bb'] = mio.import_landmark_file(
                    str(Path(path.parent.parent / 'BoundingBoxes' / (path.stem + '.pts')))
                )
                ly, lx = mp_image.landmarks['bb'].points[0]
                hy, hx = mp_image.landmarks['bb'].points[2]
                cx = (lx + hx) / 2
                cy = (ly + hy) / 2
                bb_size = int(math.ceil(max(hx - lx, hy - ly) * 4. / 6.))
                square_bb = np.array([[cy - bb_size, cx - bb_size], [cy + bb_size, cx + bb_size]])
                mp_image.landmarks['square_bb'] = PointCloud(square_bb)
                mp_image = mp_image.crop_to_landmarks_proportion(0.0, group='square_bb')
                mp_image = mp_image.resize((112, 112))

                np_image = np.expand_dims(mp_image.pixels.transpose((1, 2, 0)), 0)
                np_shape = mp_image.landmarks['PTS'].points

                prediction, = sess.run('import/add:0', feed_dict={
                    'import/input:0': np_image
                })
                assert isinstance(prediction, np.ndarray)
                prediction = prediction.reshape((68, 2))
                prediction = prediction[:, [1, 0]]
                error = normalized_batch_nme(prediction, mp_image.landmarks['PTS'].points)
                mean_error = normalized_nme(error)
                error_level = min(9, int(mean_error * 100))

                concat_image = utils.draw_landmarks_discrete(
                    np_image[0],
                    np_shape,
                    prediction
                )
                # plt.imsave('err{}/step{}.png'.format(error_level, step), concat_image)
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

            print('Errors', mean_errors.shape)
            print(
                '%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f' %
                (datetime.now(), mean_errors.mean(), auc_at_05, auc_at_08)
            )


if __name__ == '__main__':
    influence()
