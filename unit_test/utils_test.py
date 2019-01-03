import matplotlib.pyplot as plt
import menpo
import random
import sys
sys.path.append('..')

from utils import *

# =====Norm index test=====
for _ in range(1000):
    num = int(random.random() * 100)
    if num == 73:
        assert (31, 27) == norm_idx(num)
    elif num == 68:
        assert (36, 45) == norm_idx(num)
    else:
        try:
            _, _ = norm_idx(num)
            raise ValueError
        except AssertionError as e:
            pass


# =====Mirror test=====
test_image = menpo.io.import_image('../Dataset/300W/Outdoor/Images/outdoor_004.png')
assert isinstance(test_image, menpo.image.Image)
test_image.landmarks['bb'] = menpo.io.import_landmark_file(
    '../Dataset/300W/Outdoor/BoundingBoxes/outdoor_004.pts'
).bounding_box()

test_image_m = mirror_image(test_image)
assert isinstance(test_image_m, menpo.image.Image)

# Mirror bb
plt.subplot(211)
test_image.landmarks['bb_test'] = PointCloud(test_image.landmarks['bb'].points[[0, 1]])
test_image.view_landmarks(group='bb_test')
plt.subplot(212)
test_image_m.landmarks['bb_test'] = PointCloud(test_image_m.landmarks['bb'].points[[0, 1]])
test_image_m.view_landmarks(group='bb_test')
plt.show()

# Mirror 68
plt.subplot(211)
test_image.landmarks['PTS_test'] = PointCloud(test_image.landmarks['PTS'].points[range(10)])
test_image.view_landmarks(group='PTS_test')
plt.subplot(212)
test_image_m.landmarks['PTS_test'] = PointCloud(test_image_m.landmarks['PTS'].points[range(10)])
test_image_m.view_landmarks(group='PTS_test')
plt.show()

# Mirror 73
test_image = menpo.io.import_image('../Dataset/FW2/Images/t020208.png')
assert isinstance(test_image, menpo.image.Image)
test_image.landmarks['bb'] = menpo.io.import_landmark_file(
    '../Dataset/FW2/Images/t020208.pts'
).bounding_box()

test_image_m = mirror_image(test_image)
assert isinstance(test_image_m, menpo.image.Image)

plt.subplot(211)
test_image.landmarks['PTS_test'] = PointCloud(test_image.landmarks['PTS'].points[range(10)])
test_image.view_landmarks(group='PTS_test')
plt.subplot(212)
test_image_m.landmarks['PTS_test'] = PointCloud(test_image_m.landmarks['PTS'].points[range(10)])
test_image_m.view_landmarks(group='PTS_test')
plt.show()


# =====Plot test=====
test_images = menpo.io.import_images('../Dataset/FW2/Images/t02021*.png', verbose=True)
np_images = [img.pixels.transpose(1, 2, 0) for img in test_images]
np_lms = [img.landmarks['PTS'].points for img in test_images]
plot_images = batch_draw_landmarks_discrete(np_images, np_lms, np_lms)
plot_images = [
    np.concatenate((plot_images[0], plot_images[1], plot_images[2]), 1),
    np.concatenate((plot_images[3], plot_images[4], plot_images[5]), 1),
    np.concatenate((plot_images[6], plot_images[7], plot_images[8]), 1),
]
plot_images = np.concatenate((plot_images[0], plot_images[1], plot_images[2]), 0)
plt.imshow(plot_images)
plt.show()
