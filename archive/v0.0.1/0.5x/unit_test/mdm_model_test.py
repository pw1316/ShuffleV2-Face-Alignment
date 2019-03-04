import sys
sys.path.append('..')

from mdm_model import *

with tf.Graph().as_default() as graph:
    with tf.name_scope('DataProvider'):
        tf_images = tf.placeholder(tf.float32, [30, 112, 112, 3], name='in_images')
        tf_shapes = tf.placeholder(tf.float32, [30, 73, 2], name='in_shapes')
        tf_mean_shape = tf.placeholder(tf.float32, [73, 2], name='in_mean_shape')

    print('Defining model...')
    tf_model = MDMModel(
        tf_images,
        tf_shapes,
        tf_mean_shape,
        batch_size=30,
        num_patches=73,
        num_channels=3
    )
    summary_writer = tf.summary.FileWriter('../ckpt/model_test', graph)
