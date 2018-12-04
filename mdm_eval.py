"""A library to evaluate MDM on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pathlib import Path

import data_provider
import menpo
import matplotlib
import matplotlib.pyplot as plt
import mdm_model
import numpy as np
import tensorflow as tf
import time
import utils
import menpo.io as mio

# Do not use a gui toolkit for matlotlib.
matplotlib.use('Agg')

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('eval_dir', 'ckpt/eval', """Directory where to write event logs.""")
tf.flags.DEFINE_string('ckpt_dir', 'ckpt/test/', """Directory where to read model checkpoints.""")
# Flags governing the data used for the eval.
tf.flags.DEFINE_integer('num_examples', 2731, """Number of examples to run.""")
tf.flags.DEFINE_string('dataset', 'Dataset/DDETEST/Images/*.png', """The dataset path to evaluate.""")
tf.flags.DEFINE_string('device', '/cpu:0', 'the device to eval on.')
tf.flags.DEFINE_integer('batch_size', 1, """The batch size to use.""")
tf.flags.DEFINE_integer('num_patches', 68, 'Landmark number')
tf.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')
tf.flags.DEFINE_boolean('use_mirror', True, 'Use mirror evaluation')
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def plot_ced(errors, method_names=['MDM']):
    from matplotlib import pyplot as plt
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


def _eval_once(saver, summary_writer, rmse_op, summary_op):
    """Runs Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        rmse_op: rmse_op.
        summary_op: Summary op.
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
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

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = FLAGS.num_examples
            # Counts the number of correct predictions.
            errors = []

            total_sample_count = num_iter
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.dataset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                rmse = sess.run(rmse_op)
                # img_op = tf.get_default_graph().get_tensor_by_name('Network/concat:0')
                # img = sess.run(img_op)
                # print(img.shape)
                # plt.imsave('step{}.png'.format(step), img[0])
                errors.append(rmse)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = 1. / sec_per_batch
                    log_str = '{}: [{:d} batches out of {:d}] ({:.1f} examples/sec; {:.3f} sec/batch)'
                    print(log_str.format(datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()

            errors = np.vstack(errors).ravel()
            mean_rmse = errors.mean()
            auc_at_08 = (errors < .08).mean()
            auc_at_05 = (errors < .05).mean()
            ced_image = plot_ced([errors.tolist()])
            ced_plot = sess.run(tf.summary.merge([tf.summary.image('ced_plot', ced_image[None, ...])]))

            print('Errors', errors.shape)
            print(
                '%s: mean_rmse = %.4f, auc @ 0.05 = %.4f, auc @ 0.08 = %.4f [%d examples]' %
                (datetime.now(), errors.mean(), auc_at_05, auc_at_08, total_sample_count)
            )

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='AUC @ 0.08', simple_value=float(auc_at_08))
            summary.value.add(tag='AUC @ 0.05', simple_value=float(auc_at_05))
            summary.value.add(tag='Mean RMSE', simple_value=float(mean_rmse))
            summary_writer.add_summary(ced_plot, global_step)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

    
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
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        ckpt_dir = Path(FLAGS.ckpt_dir)
        reference_shape = mio.import_pickle(ckpt_dir / 'reference_shape.pkl')

        images, gt_truth, inits, _ = data_provider.batch_inputs(
                [FLAGS.dataset], reference_shape,
                batch_size=1, is_training=False)

        images_m, _, inits_m, shapes = data_provider.batch_inputs(
            [FLAGS.dataset], reference_shape,
            batch_size=1, is_training=False, mirror_image=True)

        print('Loading model...')
        with tf.device(FLAGS.device):
            model = mdm_model.MDMModel(
                images, gt_truth, inits,
                num_iterations=4,
                num_patches=FLAGS.num_patches,
                patch_shape=(FLAGS.patch_size, FLAGS.patch_size)
            )
            tf.get_variable_scope().reuse_variables()
            model_m = mdm_model.MDMModel(
                images_m, gt_truth, inits_m,
                num_iterations=4,
                num_patches=FLAGS.num_patches,
                patch_shape=(FLAGS.patch_size, FLAGS.patch_size)
            )

        if FLAGS.use_mirror:
            avg_pred = model.prediction + tf.py_func(flip_predictions, (model_m.prediction, shapes), (tf.float32, ))[0]
            avg_pred /= 2.
        else:
            avg_pred = model.prediction

        # Calculate predictions.
        norm_error = model.normalized_rmse(avg_pred, gt_truth)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)
        _eval_once(saver, summary_writer, norm_error, summary_op)


if __name__ == '__main__':
    evaluate()
