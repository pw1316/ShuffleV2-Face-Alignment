import cv2
import json
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

_mirrored_parts_68 = np.hstack([
    _jaw_indices[::-1], _right_eyebrow_indices[::-1], _left_eyebrow_indices[::-1],
    _upper_nose_indices, _lower_nose_indices[::-1],
    np.roll(_right_eye_indices[::-1], 4), np.roll(_left_eye_indices[::-1], 4),
    np.roll(_outer_mouth_indices[::-1], 7),
    np.roll(_inner_mouth_indices[::-1], 5)
])
_normalizer_68 = (36, 45)
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
_normalizer_73 = (31, 27)
# ===== 73 =====

# ===== 75 =====
_mirrored_parts_75 = np.hstack([
    _mirrored_parts_73,
    np.arange(73, 75)[::-1]
])
_normalizer_75 = (31, 27)
# ===== 75 =====

_mirrored_parts = {
    68: _mirrored_parts_68,
    73: _mirrored_parts_73,
    75: _mirrored_parts_75
}

_normalizer = {
    68: _normalizer_68,
    73: _normalizer_73,
    75: _normalizer_75
}


def norm_idx(num_patches):
    return _normalizer[num_patches]


# =====Mirror=====
def mirror_landmarks(landmarks, image_width):
    """
    Mirror landmarks along y-axis
    :param landmarks: landmarks [[y0, x0], [y1, x1], ...]
    :param image_width: width of the image, in pixels
    :return: mirrored landmarks
    """
    mirrored = np.zeros_like(landmarks, landmarks.dtype)
    tmp = np.array([0, image_width]) - landmarks
    tmp[:, 0] = -tmp[:, 0]
    mirrored[...] = tmp[_mirrored_parts[landmarks.shape[0]]]
    return mirrored


def mirror_bounding_box(bb, image_width):
    """
    Mirror bounding box along y-axis
    :param bb: bounding box [[y_min, x_min],[y_max, x_max]]
    :param image_width: width of the image, in pixels
    :return: mirrored landmarks
    """
    mirrored = np.zeros_like(bb, bb.dtype)
    mirrored[:, 0] = bb[:, 0]
    mirrored[:, 1] = (image_width - bb[:, 1])[::-1, :]
    return mirrored


def mirror_image(image, landmark_group, bb):
    mirrored_image = image[:, ::-1, :].copy()
    landmark_group = landmark_group or []
    mirrored_landmark_group = []
    for landmarks in landmark_group:
        mirrored_landmark_group.append(mirror_landmarks(landmarks, image.shape[1]))
    mirrored_bb = mirror_bounding_box(bb, image.shape[1])
    return mirrored_image, mirrored_landmark_group, mirrored_bb


# =====Plot=====
def draw_landmarks(image, ground_truth, prediction):
    image = image.copy()
    if ground_truth is not None:
        for v in ground_truth:
            cv2.circle(image, (int(v[1]), int(v[0])), 1, (1, 0, 0), -1)
    if prediction is not None:
        for v in prediction:
            cv2.circle(image, (int(v[1]), int(v[0])), 1, (1, 1, 1), -1)
    return image


def batch_draw_landmarks(images, ground_truths, predictions):
    return np.array(
        [draw_landmarks(img, gt, pr) for img, gt, pr in zip(images, ground_truths, predictions)]
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
