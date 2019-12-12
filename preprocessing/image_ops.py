import cv2
import numpy as np

from lib.dataset import resize


def find_circle(image, kernel=33):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = resize(image, 512, size_is_min=True)
    image_gray = cv2.medianBlur(image_gray, kernel)
    height, width = image_gray.shape
    min_dim = min(height, width)
    ratio = min_dim / 512
    circles = cv2.HoughCircles(image_gray,
                               cv2.HOUGH_GRADIENT,
                               1,
                               20,
                               param1=50,
                               param2=30,
                               minRadius=min_dim // 4,
                               maxRadius=0)
    if circles is None:
        return None
    circles = np.around(circles)
    # get coordinates and radius of first circle
    # x - horizontal dimension, left to right
    # y - vertical dimension, top to bottom
    x, y, r = int(circles[0, 0, 0]), int(circles[0, 0, 1]), int(circles[0, 0, 2])
    #     print(x, y, r)
    return int(x * ratio), int(y * ratio), int(r * ratio)


def crop_by_threshold(img, threshold=7):
    init_h, init_w = img.shape[0], img.shape[1]
    gray_img = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > threshold
    new_img = img[np.ix_(mask.any(1), mask.any(0))]
    new_h, new_w = new_img.shape[0], new_img.shape[1]
    if new_h == 0 or new_w == 0:
        return img
    # if any side is cropped more than 3-fold, return original image
    if init_h / new_h > 3 or init_w / new_w > 3:
        return img
    else:
        return new_img


# Ben's local brightness adjustement
# https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801#latest-370950
def bens_preprocess(image, size_is_min, new_size=512, sigmax=None):
    image = resize(image, size_is_min, new_size)
    if sigmax is None:
        sigmax = min(image.shape[0], image.shape[1]) // 60
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmax), -4, 128)
    return image
