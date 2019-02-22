import menpo.io as mio
import numpy as np
from pathlib import Path
import tensorflow as tf

import data_provider
import mdm_model
import utils

g_config = utils.load_config()


def ckpt_pb(pb_path, lite_path):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # to pb
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
        path_base = Path(g_config['eval_dataset']).parent.parent
        _mean_shape = mio.import_pickle(path_base / 'mean_shape.pkl')
        _mean_shape = data_provider.align_reference_shape_to_112(_mean_shape)
        assert isinstance(_mean_shape, np.ndarray)
        print(_mean_shape.shape)

        tf_img = tf.placeholder(dtype=tf.float32, shape=(1, 112, 112, 3), name='Inputs/InputImage')
        tf_dummy = tf.placeholder(dtype=tf.float32, shape=(1, 73, 2), name='dummy')
        tf_shape = tf.constant(_mean_shape, dtype=tf.float32, shape=(73, 2), name='Inputs/MeanShape')

        mdm_model.MDMModel(
            tf_img,
            tf_dummy,
            tf_shape,
            batch_size=1,
            num_patches=g_config['num_patches'],
            num_channels=3,
            multiplier=g_config['multiplier'],
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

    # to tflite
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='prefix',
            op_dict=None,
            producer_op_list=None
        )
        tf.summary.FileWriter('.', graph=graph)

        converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
            pb_path,
            ['Inputs/InputImage'],
            ['Network/Predict/Reshape', 'Inputs/MeanShape']
        )
        with tf.gfile.FastGFile(lite_path, mode='wb') as f:
            f.write(converter.convert())
        # Load TFLite model and allocate tensors.
        interpreter = tf.contrib.lite.Interpreter(model_path=lite_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)


ckpt_pb('shuffle.pb', 'shuffle.tflite')
