import cv2
import json
from menpo.shape import PointCloud
import numpy as np
import tensorflow as tf

# ===== 68 =====
_jaw_indices = np.arange(0, 17)
_left_eyebrow_indices = np.arange(17, 22)
_right_eyebrow_indices = np.arange(22, 27)
_upper_nose_indices = np.arange(27, 31)
_lower_nose_indices = np.arange(31, 36)
_left_eye_indices = np.arange(36, 42)
_right_eye_indices = np.arange(42, 48)
_outer_mouth_indices = np.arange(48, 60)
_inner_mouth_indices = np.arange(60, 68)

_parts_68 = (_jaw_indices, _left_eyebrow_indices, _right_eyebrow_indices, _upper_nose_indices,
             _lower_nose_indices, _left_eye_indices, _right_eye_indices,
             _outer_mouth_indices, _inner_mouth_indices)

_mirrored_parts_68 = np.hstack([
    _jaw_indices[::-1], _right_eyebrow_indices[::-1], _left_eyebrow_indices[::-1],
    _upper_nose_indices, _lower_nose_indices[::-1],
    np.roll(_right_eye_indices[::-1], 4), np.roll(_left_eye_indices[::-1], 4),
    np.roll(_outer_mouth_indices[::-1], 7),
    np.roll(_inner_mouth_indices[::-1], 5)
])
# ===== 68 =====

# ===== 73 =====
_jaw_73 = np.arange(0, 15)
_left_eyebrow_73 = np.arange(15, 21)
_right_eyebrow_73 = np.arange(21, 27)
_left_eye_73 = np.arange(31, 35)
_right_eye_73 = np.arange(27, 31)
_nose_73 = np.arange(35, 44)
_nose_hole_73 = np.arange(44, 46)
_nose_point_73 = np.arange(64, 65)
_upper_outer_mouth_73 = np.arange(46, 53)
_lower_outer_mouth_73 = np.arange(53, 58)
_upper_inner_mouth_73 = np.arange(61, 64)
_lower_inner_mouth_73 = np.arange(58, 61)
_eye_extra = np.arange(65, 73)

_parts_73 = (
    _jaw_73,
    _left_eyebrow_73,
    _right_eyebrow_73,
    _right_eye_73,
    _left_eye_73,
    _nose_73,
    _nose_hole_73,
    _upper_outer_mouth_73,
    _lower_outer_mouth_73,
    _lower_inner_mouth_73,
    _upper_inner_mouth_73,
    _nose_point_73,
    _eye_extra
)

_mirrored_parts_73 = np.hstack([
    _jaw_73[::-1],
    _right_eyebrow_73,
    _left_eyebrow_73,
    _left_eye_73,
    _right_eye_73,
    _nose_73[::-1],
    _nose_hole_73[::-1],
    _upper_outer_mouth_73[::-1],
    _lower_outer_mouth_73[::-1],
    _lower_inner_mouth_73[::-1],
    _upper_inner_mouth_73[::-1],
    _nose_point_73,
    _eye_extra[::-1]
])
# ===== 73 =====


def norm_idx(num_patches):
    if num_patches == 68:
        return 36, 45
    elif num_patches == 73:
        return 31, 27
    assert False


# =====Mirror=====
def _mirror_landmarks(landmarks, image_size, mirrored_parts):
    tmp = np.array([0, image_size[1]]) - landmarks.as_vector().reshape(-1, 2)
    tmp[:, 0] = -tmp[:, 0]
    return PointCloud(tmp[mirrored_parts])


def mirror_landmarks_68(lms, image_size):
    return _mirror_landmarks(lms, image_size, _mirrored_parts_68)


def mirror_landmarks_73(lms, image_size):
    return _mirror_landmarks(lms, image_size, _mirrored_parts_73)


def mirror_bounding_box(lms, image_size):
    return _mirror_landmarks(lms, image_size, range(4)).bounding_box()


def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:
        if group == 'PTS':
            if im.landmarks[group].points.shape[0] == 68:
                im.landmarks[group] = mirror_landmarks_68(im.landmarks[group], im.shape)
            elif im.landmarks[group].points.shape[0] == 73:
                im.landmarks[group] = mirror_landmarks_73(im.landmarks[group], im.shape)
        elif group == 'bb':
            im.landmarks[group] = mirror_bounding_box(im.landmarks[group], im.shape)
    return im


# =====Plot=====
def draw_landmarks_discrete(image, ground_truth, prediction):
    image = image.copy()
    if ground_truth is not None:
        for v in ground_truth:
            cv2.circle(image, (int(v[1]), int(v[0])), 1, (1, 0, 0), -1)
    if prediction is not None:
        for v in prediction:
            cv2.circle(image, (int(v[1]), int(v[0])), 1, (1, 1, 1), -1)
    return image


def batch_draw_landmarks_discrete(images, ground_truths, predictions):
    return np.array(
        [draw_landmarks_discrete(img, gt, pr) for img, gt, pr in zip(images, ground_truths, predictions)]
    )


# =====Config=====
def load_config():
    if 'c' not in tf.flags.FLAGS:
        tf.flags.DEFINE_string('c', 'config.json', """Model config file""")
    with open(tf.flags.FLAGS.c, 'r') as g_config:
        g_config = json.load(g_config)
    for k in g_config:
        print('%s:' % k, g_config[k], type(g_config[k]))
    assert isinstance(g_config, dict)
    res = input('OK?(Y/N): ')
    return g_config if res == 'y' or res == 'Y' else None
