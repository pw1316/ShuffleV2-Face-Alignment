from functools import partial
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path

import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import os
import tensorflow as tf
import detect
import utils
import random

FLAGS = tf.flags.FLAGS


def build_reference_shape(paths, diagonal=200):
    """Builds the reference shape.

    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()
        landmarks += [
            group
            for group in mio.import_landmark_files(path, verbose=True)
            if group.n_points == FLAGS.num_patches
        ]

    return compute_reference_shape(landmarks, diagonal=diagonal).points.astype(np.float32)


def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale

    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im


def align_reference_shape(reference_shape, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))
    with tf.name_scope('align_shape_to_bb', values=[reference_shape, bb]):
        min_xy = tf.reduce_min(reference_shape, 0)
        max_xy = tf.reduce_max(reference_shape, 0)
        min_x, min_y = min_xy[0], min_xy[1]
        max_x, max_y = max_xy[0], max_xy[1]
        reference_shape_bb = tf.stack([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        ratio = norm(bb) / norm(reference_shape_bb)
        initial_shape = tf.add(
            (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
            tf.reduce_mean(bb, 0),
            name='initial_shape'
        )
    return initial_shape


def random_shape(tf_shape, tf_mean_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.

    Args:
      tf_shape: a numpy array [num_landmarks, 2]
      tf_mean_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    with tf.name_scope('random_initial_shape', values=[tf_shape, tf_mean_shape]):
        tf_random_bb, = tf.py_func(
            synthesize, [tf_shape], [tf.float32],
            stateful=True,
            name='random_bb'
        )  # Random bb for shape
        tf_random_shape = align_reference_shape(tf_mean_shape, tf_random_bb)  # align mean shape to bb
        tf_random_shape.set_shape(tf_mean_shape.get_shape())
    return tf_random_shape


def get_noisy_init_from_bb(reference_shape, bb, noise_percentage=.02):
    """Roughly aligns a reference shape to a bounding box.

    This adds some uniform noise for translation and scale to the
    aligned shape.

    Args:
      reference_shape: a numpy array [num_landmarks, 2]
      bb: bounding box, a numpy array [4, ]
      noise_percentage: noise presentation to add.
    Returns:
      The aligned shape, as a numpy array [num_landmarks, 2]
    """
    bb = PointCloud(bb)
    reference_shape = PointCloud(reference_shape)

    bb = noisy_shape_from_bounding_box(
        reference_shape,
        bb,
        noise_percentage=[noise_percentage, 0, noise_percentage]).bounding_box(
        )

    return align_shape_with_bounding_box(reference_shape, bb).points


def prepare_images(paths, group=None, verbose=True):
    """Save Train Images to TFRecord
    Args:
        paths: a list of strings containing the data directories.
        group: landmark group containing the ground truth landmarks.
        verbose: boolean, print debugging info.
    Returns:
        None
    """
    train_dir = Path(FLAGS.train_dir)
    image_paths = []
    reference_shape = PointCloud(build_reference_shape(paths))
    mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    print('created reference_shape.pkl')

    image_shape = [0, 0, 3]  # [H, W, C]
    with tf.io.TFRecordWriter('pca.bin') as ofs:
        for path in paths:
            if verbose:
                print('Importing data from {}'.format(path))

            for image in mio.import_images(path, verbose=verbose, as_generator=True):
                group = group or image.landmarks.group_labels[0]

                bb_root = image.path.parent.parent
                try:
                    landmark = mio.import_landmark_file(
                        str(Path(bb_root / 'BoundingBoxes' / (image.path.stem + '.pts')))
                    )
                except ValueError:
                    print('skip')
                    continue
                image.landmarks['bb'] = landmark
                image = image.crop_to_landmarks_proportion(0.3, group='bb')
                image = image.rescale_to_pointcloud(reference_shape, group=group)
                image = grey_to_rgb(image)
                assert(image.pixels.shape[0] == image_shape[2])
                image_shape[0] = max(image.pixels.shape[1], image_shape[0])
                image_shape[1] = max(image.pixels.shape[2], image_shape[1])
                image_paths.append(image.path)
                features = tf.train.Features(
                    feature={
                        'pca/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.landmarks[group].points.flatten())
                        ),
                        'pca/bb': tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.landmarks['bb'].points.flatten())
                        ),
                    }
                )
                ofs.write(tf.train.Example(features=features).SerializeToString())

    # Random shuffle
    random.shuffle(image_paths)

    print('padded shape', image_shape)
    with tf.io.TFRecordWriter('train.bin') as ofs:
        print('Preparing train data...')
        for image_path in image_paths:
            print('\rPreparing {}'.format(image_path), end='     ')
            image = mio.import_image(image_path)
            group = group or image.landmarks.group_labels[0]
            bb_root = image.path.parent.parent
            image.landmarks['bb'] = mio.import_landmark_file(
                str(Path(bb_root / 'BoundingBoxes' / (image.path.stem + '.pts')))
            )
            image = image.crop_to_landmarks_proportion(0.3, group='bb')
            image = image.rescale_to_pointcloud(reference_shape, group=group)
            image = grey_to_rgb(image)

            # Padding to the same size
            padded_image = np.random.rand(*image_shape).astype(np.float32)
            height, width = image.pixels.shape[1:]  # [C, H, W]
            dy = max(int((image_shape[0] - height - 1) / 2), 0)
            dx = max(int((image_shape[1] - width - 1) / 2), 0)
            padded_landmark = image.landmarks[group].points
            padded_landmark[:, 0] += dy
            padded_landmark[:, 1] += dx
            padded_image[dy:(height + dy), dx:(width + dx), :] = image.pixels.transpose(1, 2, 0)
            features = tf.train.Features(
                feature={
                    'train/image': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(padded_image.tostring())])
                    ),
                    'train/shape': tf.train.Feature(
                        float_list=tf.train.FloatList(value=padded_landmark.flatten())
                    )
                }
            )
            ofs.write(tf.train.Example(features=features).SerializeToString())
        print('')

    with open('meta.txt', 'w') as ofs:
        for s in image_shape[:-1]:
            ofs.write('{} '.format(s))
        ofs.write('{}'.format(image_shape[-1]))


def load_images(paths, group=None, verbose=True):
    """Loads and rescales input images to the diagonal of the reference shape.

    Args:
        paths: a list of strings containing the data directories.
        group: landmark group containing the ground truth landmarks.
        verbose: boolean, print debugging info.
    Returns:
        images: a list of numpy arrays containing images.
        shapes: a list of the ground truth landmarks.
        reference_shape: a numpy array [num_landmarks, 2].
        shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    bbs = []

    reference_shape = PointCloud(build_reference_shape(paths))

    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            group = group or im.landmarks.group_labels[0]

            bb_root = im.path.parent.parent
            try:
                lms = mio.import_landmark_file(str(Path(bb_root / 'BoundingBoxes' / (im.path.stem + '.pts'))))
            except ValueError:
                print('skip')
                continue
            im.landmarks['bb'] = lms
            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            im = im.rescale_to_pointcloud(reference_shape, group=group)
            im = grey_to_rgb(im)
            images.append(im.pixels.transpose(1, 2, 0))
            shapes.append(im.landmarks[group])
            bbs.append(im.landmarks['bb'])

    train_dir = Path(FLAGS.train_dir)
    mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    print('created reference_shape.pkl using the {} group'.format(group))

    pca_model = detect.create_generator(shapes, bbs)

    # Pad images to max length
    max_shape = np.max([im.shape for im in images], axis=0)
    max_shape = [len(images)] + list(max_shape)
    padded_images = np.random.rand(*max_shape).astype(np.float32)
    print(padded_images.shape)

    for i, im in enumerate(images):
        height, width = im.shape[:2]
        dy = max(int((max_shape[1] - height - 1) / 2), 0)
        dx = max(int((max_shape[2] - width - 1) / 2), 0)
        lms = shapes[i]
        pts = lms.points
        pts[:, 0] += dy
        pts[:, 1] += dx
        padded_images[i, dy:(height+dy), dx:(width+dx)] = im

    return padded_images, shapes, reference_shape.points, pca_model


def load_image(path, reference_shape, is_training=False, group='PTS',
               mirror_image=False):
    """Load an annotated image.

    In the directory of the provided image file, there
    should exist a landmark file (.pts) with the same
    basename as the image file.

    Args:
      path: a path containing an image file.
      reference_shape: a numpy array [num_landmarks, 2]
      is_training: whether in training mode or not.
      group: landmark group containing the grounth truth landmarks.
      mirror_image: flips horizontally the image's pixels and landmarks.
    Returns:
      pixels: a numpy array [width, height, 3].
      estimate: an initial estimate a numpy array [num_landmarks, 2].
      gt_truth: the ground truth landmarks, a numpy array [num_landmarks, 2].
    """
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    assert(isinstance(path, str))
    im = mio.import_image(path)
    bb_root = im.path.parent.parent

    im.landmarks['bb'] = mio.import_landmark_file(str(Path(bb_root / 'BoundingBoxes' / (im.path.stem + '.pts'))))

    im = im.crop_to_landmarks_proportion(0.3, group='bb')
    reference_shape = PointCloud(reference_shape)
    bb = im.landmarks['bb'].bounding_box()
    im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape, bb)
    im = im.rescale_to_pointcloud(reference_shape, group='__initial')

    if mirror_image:
        im = utils.mirror_image(im)

    lms = im.landmarks[group]
    initial = im.landmarks['__initial']

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)
    return pixels.astype(np.float32).copy(), gt_truth, estimate


def distort_color(image, thread_id=0, stddev=0.1):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      stddev: gaussian noise dev
    Returns:
      color-distorted image
    """
    with tf.name_scope('distort_color', values=[image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
                tf.shape(image),
                stddev=stddev,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def batch_inputs(paths,
                 reference_shape,
                 batch_size=32,
                 is_training=False,
                 mirror_image=False):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_training: whether in training mode.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, num_landmarks, 2].
      lms_init: a tf tensor of shape [batch_size, num_landmarks, 2].
    """
    _files = [list(map(str, sorted(Path(d).parent.glob(Path(d).name)))) for d in paths]
    for i, _ in enumerate(_files):
        _files[i] = list(filter(lambda x: os.path.exists(x[:-4] + '.pts'), _files[i]))
    files = tf.concat(_files, 0)

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=1000)

    filename = filename_queue.dequeue()

    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training,
                mirror_image=mirror_image),
        [filename, reference_shape],  # input arguments
        [tf.float32, tf.float32, tf.float32],  # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, reference_shape.shape)
    lms_init = tf.reshape(lms_init, reference_shape.shape)

    images, lms, inits, shapes = tf.train.batch(
                                    [image, lms, lms_init, tf.shape(image)],
                                    batch_size=batch_size,
                                    num_threads=4 if is_training else 1,
                                    capacity=1000,
                                    enqueue_many=False,
                                    dynamic_pad=True)

    return images, lms, inits, shapes
