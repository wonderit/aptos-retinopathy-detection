from os import makedirs
from os.path import isdir

import cv2
import pandas as pd


# slightly modified function from
# https://www.kaggle.com/benjaminwarner/starter-code-resized-15-19-blindness-images
def resize_images(location, name, extension, save_location, desired_size=1024):
    img = cv2.imread(f"{location}/{name}.{extension}")
    if not isdir(save_location):
        makedirs(save_location)

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(gray,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours)

    if w > 200 and h > 200:
        new_img = img[y:y + h, x:x + w]
        height, width, _ = new_img.shape

        if max([height, width]) > desired_size:
            ratio = float(desired_size / max([height, width]))
            new_img = cv2.resize(new_img,
                                 tuple([int(width * ratio), int(height * ratio)]),
                                 interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(f'{save_location}/{name}.jpg', new_img)
    else:
        print(f'No bounding for {name}')
        cv2.imwrite(f'{save_location}/{name}.jpg', img)


if __name__ == "__main__":
    IMG_PATH_SOURCE = r"/path/to/messidor1/image/folder"
    IMG_PATH_DESTINATION = r"../input/messidor1_jpg"

    DF_PATH_MESSIDOR = r"../input/messidor1_labels_adjudicated.csv"

    df_messidor = pd.read_csv(DF_PATH_MESSIDOR)
    df_messidor = df_messidor[df_messidor.adjudicated_dr_grade > -1]
    X_messidor = df_messidor.image.values

    for img in X_messidor:
        resize_images(IMG_PATH_SOURCE, img, "tif", IMG_PATH_DESTINATION)
