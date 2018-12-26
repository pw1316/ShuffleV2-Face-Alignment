from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path

import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import detect
import random


def build_reference_shape(paths, num_patches=73, diagonal=200):
    """Builds the reference shape.

    Args:
        paths: train image paths.
        num_patches: number of landmarks
        diagonal: the diagonal of the reference shape in pixels.
    Returns:
        the reference shape.
    """
    landmarks = []
    for path in paths:
        group = mio.import_landmark_file(path.parent / (path.stem + '.pts'))
        if group.n_points == num_patches:
            landmarks += [group]
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


def align_reference_shape_to_112(reference_shape):
    assert isinstance(reference_shape, np.ndarray)

    def norm(x):
        return np.sqrt(np.sum(np.square(x - np.mean(x, 0))))
    min_xy = np.min(reference_shape, 0)
    max_xy = np.max(reference_shape, 0)
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]
    reference_shape_bb = np.vstack([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    bb = np.vstack([[0.0, 0.0], [112.0, 0.0], [112.0, 112.0], [0.0, 112.0]])
    ratio = norm(bb) / norm(reference_shape_bb)
    reference_shape = (reference_shape - np.mean(reference_shape_bb, 0)) * ratio + np.mean(bb, 0)
    return reference_shape


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


def prepare_images(paths, num_patches=73, verbose=True):
    """Save Train Images to TFRecord
    Args:
        paths: a list of strings containing the data directories.
        num_patches: number of landmarks
        verbose: boolean, print debugging info.
    Returns:
        None
    """
    if len(paths) == 0:
        return
    # .../<Dataset>/Images/*.png -> .../<Dataset>
    path_base = Path(paths[0]).parent.parent
    image_paths = []

    # First: get all image paths
    for path in paths:
        for file in Path('.').glob(path):
            try:
                mio.import_landmark_file(
                    str(Path(file.parent.parent / 'BoundingBoxes' / (file.stem + '.pts')))
                )
            except ValueError:
                continue
            image_paths.append(file)
    print('Got all image paths...')

    # Second: split to train, test and validate. 7:2:1
    if Path(path_base / 'train_img.txt').exists():
        with Path(path_base / 'train_img.txt').open('rb') as train_ifs, \
                Path(path_base / 'test_img.txt').open('rb') as test_ifs, \
                Path(path_base / 'val_img.txt').open('rb') as val_ifs:
            train_paths = [Path(line[:-1].decode('utf-8')) for line in train_ifs.readlines()]
            test_paths = [Path(line[:-1].decode('utf-8')) for line in test_ifs.readlines()]
            val_paths = [Path(line[:-1].decode('utf-8')) for line in val_ifs.readlines()]
    else:
        random.shuffle(image_paths)
        num_train = int(len(image_paths) * 0.7)
        num_test = int(len(image_paths) * 0.2)
        train_paths = sorted(image_paths[:num_train])
        test_paths = sorted(image_paths[num_train:num_train+num_test])
        val_paths = sorted(image_paths[num_train+num_test:])
        with Path(path_base / 'train_img.txt').open('wb') as train_ofs, \
                Path(path_base / 'test_img.txt').open('wb') as test_ofs, \
                Path(path_base / 'val_img.txt').open('wb') as val_ofs:
            train_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in train_paths])
            test_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in test_paths])
            val_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in val_paths])
    print('Found Train/Test/Validate {}/{}/{}'.format(len(train_paths), len(test_paths), len(val_paths)))

    # Third: export reference shape on train
    if Path(path_base / 'reference_shape.pkl').exists():
        reference_shape = PointCloud(mio.import_pickle(path_base / 'reference_shape.pkl'))
    else:
        reference_shape = PointCloud(build_reference_shape(train_paths, num_patches))
        mio.export_pickle(reference_shape.points, path_base / 'reference_shape.pkl', overwrite=True)
    print('Created reference_shape.pkl')

    # Fourth: image shape & pca
    image_shape = [0, 0, 3]  # [H, W, C]
    if Path(path_base / 'pca.bin').exists() and Path(path_base / 'meta.txt').exists():
        with Path(path_base / 'meta.txt').open('r') as ifs:
            image_shape = [int(x) for x in ifs.read().split(' ')]
    else:
        with tf.io.TFRecordWriter(str(path_base / 'pca.bin')) as ofs:
            counter = 0
            for path in train_paths:
                counter += 1
                if verbose:
                    status = 10.0 * counter / len(train_paths)
                    status_str = '\rPreparing {:2.2f}%['.format(status * 10)
                    for i in range(int(status)):
                        status_str += '='
                    for i in range(int(status), 10):
                        status_str += ' '
                    status_str += '] {}     '.format(path)
                    print(status_str, end='')
                mp_image = mio.import_image(path)
                mp_image.landmarks['bb'] = mio.import_landmark_file(
                    str(Path(mp_image.path.parent.parent / 'BoundingBoxes' / (mp_image.path.stem + '.pts')))
                )
                mp_image = mp_image.crop_to_landmarks_proportion(0.3, group='bb')
                mp_image = mp_image.rescale_to_pointcloud(reference_shape, group='PTS')
                mp_image = grey_to_rgb(mp_image)
                assert(mp_image.pixels.shape[0] == image_shape[2])
                image_shape[0] = max(mp_image.pixels.shape[1], image_shape[0])
                image_shape[1] = max(mp_image.pixels.shape[2], image_shape[1])
                features = tf.train.Features(
                    feature={
                        'pca/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=mp_image.landmarks['PTS'].points.flatten())
                        ),
                        'pca/bb': tf.train.Feature(
                            float_list=tf.train.FloatList(value=mp_image.landmarks['bb'].points.flatten())
                        ),
                    }
                )
                ofs.write(tf.train.Example(features=features).SerializeToString())
            if verbose:
                print('')
        with Path(path_base / 'meta.txt').open('w') as ofs:
            for s in image_shape[:-1]:
                ofs.write('{} '.format(s))
            ofs.write('{}'.format(image_shape[-1]))
    print('Image shape', image_shape)

    # Fifth: train data
    if Path(path_base / 'train.bin').exists():
        pass
    else:
        random.shuffle(train_paths)
        with tf.io.TFRecordWriter(str(path_base / 'train.bin')) as ofs:
            print('Preparing train data...')
            counter = 0
            for path in train_paths:
                counter += 1
                if verbose:
                    status = 10.0 * counter / len(train_paths)
                    status_str = '\rPreparing {:2.2f}%['.format(status * 10)
                    for i in range(int(status)):
                        status_str += '='
                    for i in range(int(status), 10):
                        status_str += ' '
                    status_str += '] {}     '.format(path)
                    print(status_str, end='')
                mp_image = mio.import_image(path)
                mp_image.landmarks['bb'] = mio.import_landmark_file(
                    str(Path(mp_image.path.parent.parent / 'BoundingBoxes' / (mp_image.path.stem + '.pts')))
                )
                mp_image = mp_image.crop_to_landmarks_proportion(0.3, group='bb')
                mp_image = mp_image.rescale_to_pointcloud(reference_shape, group='PTS')
                mp_image = grey_to_rgb(mp_image)
                # Padding to the same size
                height, width = mp_image.pixels.shape[1:]  # [C, H, W]
                dy = max(int((image_shape[0] - height - 1) / 2), 0)
                dx = max(int((image_shape[1] - width - 1) / 2), 0)
                padded_image = np.random.rand(*image_shape).astype(np.float32)
                padded_image[dy:(height + dy), dx:(width + dx), :] = mp_image.pixels.transpose(1, 2, 0)
                padded_landmark = mp_image.landmarks['PTS'].points
                padded_landmark[:, 0] += dy
                padded_landmark[:, 1] += dx
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
            if verbose:
                print('')

    # Sixth: test data
    if Path(path_base / 'test.bin').exists():
        pass
    else:
        with tf.io.TFRecordWriter(str(path_base / 'test.bin')) as ofs:
            print('Preparing test data...')
            counter = 0
            for path in test_paths:
                counter += 1
                if verbose:
                    status = 10.0 * counter / len(test_paths)
                    status_str = '\rPreparing {:2.2f}%['.format(status * 10)
                    for i in range(int(status)):
                        status_str += '='
                    for i in range(int(status), 10):
                        status_str += ' '
                    status_str += '] {}     '.format(path)
                    print(status_str, end='')
                mp_image = mio.import_image(path)
                mp_image.landmarks['bb'] = mio.import_landmark_file(
                    str(Path(mp_image.path.parent.parent / 'BoundingBoxes' / (mp_image.path.stem + '.pts')))
                )
                mp_image = mp_image.crop_to_landmarks_proportion(0.3, group='bb')
                mp_bb = mp_image.landmarks['bb'].bounding_box()
                mp_image.landmarks['init'] = align_shape_with_bounding_box(reference_shape, mp_bb)
                mp_image = mp_image.rescale_to_pointcloud(reference_shape, group='init')
                mp_image = grey_to_rgb(mp_image)
                # Padding to the same size
                height, width = mp_image.pixels.shape[1:]  # [C, H, W]
                dy = max(int((256 - height - 1) / 2), 0)  # 200*(1+0.3*2)/sqrt(2) == 226.7
                dx = max(int((256 - width - 1) / 2), 0)  # 200*(1+0.3*2)/sqrt(2) == 226.7
                padded_image = np.random.rand(256, 256, 3).astype(np.float32)
                padded_image[dy:(height + dy), dx:(width + dx), :] = mp_image.pixels.transpose(1, 2, 0)
                padded_landmark = mp_image.landmarks['PTS'].points
                padded_landmark[:, 0] += dy
                padded_landmark[:, 1] += dx
                padded_init_landmark = mp_image.landmarks['init'].points
                padded_init_landmark[:, 0] += dy
                padded_init_landmark[:, 1] += dx
                features = tf.train.Features(
                    feature={
                        'test/image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[tf.compat.as_bytes(padded_image.tostring())])
                        ),
                        'test/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=padded_landmark.flatten())
                        ),
                        'test/init': tf.train.Feature(
                            float_list=tf.train.FloatList(value=padded_init_landmark.flatten())
                        )
                    }
                )
                ofs.write(tf.train.Example(features=features).SerializeToString())
            if verbose:
                print('')


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
