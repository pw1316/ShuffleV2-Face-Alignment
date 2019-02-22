import menpo.shape as mshape
from menpofit.builder import compute_reference_shape
from pathlib import Path

import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import random


def build_mean_shape(paths, num_patches):
    landmarks = []
    for path in paths:
        landmark_path = path.parent.parent / 'Fix3' / (path.stem + '.txt')
        landmark = np.genfromtxt(landmark_path)[:, [1, 0]]
        if landmark.shape[0] == num_patches:
            landmarks += [mshape.PointCloud(landmark)]
    return compute_reference_shape(landmarks, verbose=True).points.astype(np.float32)


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


def load_image(path, proportion, size):
    mp_image = mio.import_image(path)
    assert isinstance(mp_image, menpo.image.Image)

    miny, minx = np.min(mp_image.landmarks['PTS'].bounding_box().points, 0)
    maxy, maxx = np.max(mp_image.landmarks['PTS'].bounding_box().points, 0)
    bbsize = max(maxx - minx, maxy - miny)

    pady = int(max(max(bbsize * proportion - miny, 0), max(maxy + bbsize * proportion - mp_image.height, 0))) + 100
    padx = int(max(max(bbsize * proportion - minx, 0), max(maxx + bbsize * proportion - mp_image.width, 0))) + 100

    c, h, w = mp_image.pixels.shape
    pad_image = np.random.rand(c, h + pady + pady, w + padx + padx)
    pad_image[:, pady: pady + h, padx: padx + w] = mp_image.pixels
    pad_shape = mp_image.landmarks['PTS'].points + np.array([pady, padx])

    mp_image = menpo.image.Image(pad_image)
    mp_image.landmarks['PTS'] = PointCloud(pad_shape)
    assert isinstance(mp_image, menpo.image.Image)

    miny, minx = np.min(mp_image.landmarks['PTS'].bounding_box().points, 0)
    maxy, maxx = np.max(mp_image.landmarks['PTS'].bounding_box().points, 0)
    bbsize = max(maxx - minx, maxy - miny)

    center = [(miny + maxy) / 2., (minx + maxx) / 2.]
    mp_image.landmarks['bb'] = PointCloud(
        [
            [center[0] - bbsize * 0.5, center[1] - bbsize * 0.5],
            [center[0] + bbsize * 0.5, center[1] + bbsize * 0.5],
        ]
    ).bounding_box()

    mp_image = mp_image.crop_to_landmarks_proportion(proportion, group='bb', constrain_to_boundary=False)
    assert isinstance(mp_image, menpo.image.Image)

    mp_image = mp_image.resize((size, size))
    assert isinstance(mp_image, menpo.image.Image)

    mp_image = grey_to_rgb(mp_image)
    assert isinstance(mp_image, menpo.image.Image)
    return mp_image


def prepare_images(paths, num_patches, verbose=True):
    """Save Train/Test/Validate Images to TFRecord, for ShuffleNet
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

    # First & Second: get all image paths; split to train, test and validate. 7:2:1
    if Path(path_base / 'train_img.txt').exists():
        with Path(path_base / 'train_img.txt').open('rb') as train_ifs, \
                Path(path_base / 'test_img.txt').open('rb') as test_ifs, \
                Path(path_base / 'val_img.txt').open('rb') as val_ifs:
            train_paths = [Path(line[:-1].decode('utf-8')) for line in train_ifs.readlines()]
            test_paths = [Path(line[:-1].decode('utf-8')) for line in test_ifs.readlines()]
            val_paths = [Path(line[:-1].decode('utf-8')) for line in val_ifs.readlines()]
        print('Found Train/Test/Validate {}/{}/{}'.format(len(train_paths), len(test_paths), len(val_paths)))
    else:
        for path in paths:
            for file in Path('.').glob(path):
                image_paths.append(file)
        print('Got all image paths...')
        random.shuffle(image_paths)
        num_train = int(len(image_paths) * 0.9)
        num_test = int(len(image_paths) * 0.09)
        train_paths = sorted(image_paths[:num_train])
        test_paths = sorted(image_paths[num_train:num_train+num_test])
        val_paths = sorted(image_paths[num_train+num_test:])
        with Path(path_base / 'train_img.txt').open('wb') as train_ofs, \
                Path(path_base / 'test_img.txt').open('wb') as test_ofs, \
                Path(path_base / 'val_img.txt').open('wb') as val_ofs:
            train_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in train_paths])
            test_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in test_paths])
            val_ofs.writelines([str(line).encode('utf-8') + b'\n' for line in val_paths])
        print('Write Train/Test/Validate {}/{}/{}'.format(len(train_paths), len(test_paths), len(val_paths)))

    # Third: export reference shape on train
    if Path(path_base / 'mean_shape.pkl').exists():
        mean_shape = mshape.PointCloud(mio.import_pickle(path_base / 'mean_shape.pkl'))
        print('Imported mean_shape.pkl')
    else:
        mean_shape = mshape.PointCloud(build_mean_shape(train_paths, num_patches))
        mio.export_pickle(mean_shape.points, path_base / 'mean_shape.pkl', overwrite=True)
        print('Created mean_shape.pkl')

    # Fourth: image shape & pca
    # No need for ShuffleNet

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
                mp_image = load_image(path, 0.7, 336)

                image = mp_image.pixels.transpose(1, 2, 0).astype(np.float32)
                shape = mp_image.landmarks['PTS'].points
                features = tf.train.Features(
                    feature={
                        'train/image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image.tostring())])
                        ),
                        'train/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=shape.flatten())
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

                mp_image = load_image(path, 1. / 6., 112)
                mp_image.landmarks['init'] = PointCloud(align_reference_shape_to_112(reference_shape.points))

                image = mp_image.pixels.transpose(1, 2, 0).astype(np.float32)
                shape = mp_image.landmarks['PTS'].points
                init = mp_image.landmarks['init'].points
                features = tf.train.Features(
                    feature={
                        'test/image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[tf.compat.as_bytes(image.tostring())])
                        ),
                        'test/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=shape.flatten())
                        ),
                        'test/init': tf.train.Feature(
                            float_list=tf.train.FloatList(value=init.flatten())
                        )
                    }
                )
                ofs.write(tf.train.Example(features=features).SerializeToString())
            if verbose:
                print('')

    # Seven: validate data
    if Path(path_base / 'validate.bin').exists():
        pass
    else:
        random.shuffle(val_paths)
        with tf.io.TFRecordWriter(str(path_base / 'validate.bin')) as ofs:
            print('Preparing validate data...')
            counter = 0
            for path in val_paths:
                counter += 1
                if verbose:
                    status = 10.0 * counter / len(val_paths)
                    status_str = '\rPreparing {:2.2f}%['.format(status * 10)
                    for i in range(int(status)):
                        status_str += '='
                    for i in range(int(status), 10):
                        status_str += ' '
                    status_str += '] {}     '.format(path)
                    print(status_str, end='')

                mp_image = load_image(path, 1. / 6., 112)

                image = mp_image.pixels.transpose(1, 2, 0).astype(np.float32)
                shape = mp_image.landmarks['PTS'].points
                features = tf.train.Features(
                    feature={
                        'validate/image': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[tf.compat.as_bytes(image.tostring())])
                        ),
                        'validate/shape': tf.train.Feature(
                            float_list=tf.train.FloatList(value=shape.flatten())
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
