import menpo.io as mio
import numpy as np
from pathlib import Path
import tensorflow as tf

import data_provider
import mdm_model
import utils

g_config = utils.load_config()


def ckpt_pb(pb_path):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        path_base = Path(g_config['eval_dataset']).parent.parent
        _mean_shape = mio.import_pickle(path_base / 'reference_shape.pkl')
        _mean_shape = data_provider.align_reference_shape_to_112(_mean_shape)
        assert isinstance(_mean_shape, np.ndarray)
        print(_mean_shape.shape)

        tf_img = tf.placeholder(dtype=tf.float32, shape=(1, 112, 112, 3), name='inputs/input_img')
        tf_dummy = tf.placeholder(dtype=tf.float32, shape=(1, 73, 2), name='inputs/input_shape')
        tf_shape = tf.constant(_mean_shape, dtype=tf.float32, shape=(73, 2), name='MeanShape')

        model = mdm_model.MDMModel(
            tf_img,
            tf_dummy,
            tf_shape,
            batch_size=1,
            num_patches=g_config['num_patches'],
            num_channels=3,
            is_training=False
        )

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(g_config['train_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from {} at step={}.'.format(ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['Network/Predict/add']
        )

        with tf.gfile.FastGFile(pb_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


ckpt_pb('shuffle.pb')
