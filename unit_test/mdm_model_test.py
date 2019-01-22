import sys
sys.path.append('..')

from mdm_model import *

with tf.Graph().as_default() as graph, tf.Session(graph=graph) as sess:
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
        num_channels=3,
        multiplier=0.9,
    )
    with tf.name_scope('Validate', values=[tf_images, tf_shapes, tf_mean_shape]):
        tf_model_v = MDMModel(
            tf_images,
            tf_shapes,
            tf_mean_shape,
            batch_size=30,
            num_patches=73,
            num_channels=3,
            multiplier=0.9,
            is_training=False
        )

    saver = tf.train.Saver()
    start_step = 0
    ckpt = tf.train.get_checkpoint_state('ckpt/shuffle_v2_1x')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
    summary_writer = tf.summary.FileWriter('../ckpt/model_test', graph)
