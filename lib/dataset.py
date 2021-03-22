import cv2

try:
    import jpeg4py

    _JPEG4PY = True
except:
    _JPEG4PY = False

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from torch.utils.data import Dataset


def resize(image, new_size, size_is_min, interpolation=cv2.INTER_CUBIC):
    (h, w) = image.shape[:2]
    if isinstance(new_size, int):
        if size_is_min:
            condition = h > w
        else:
            condition = h <= w
        if condition:
            new_size = (new_size, int(new_size * h / w))
        else:
            new_size = (int(new_size * w / h), new_size)
    return cv2.resize(image, new_size, interpolation)


# read original (full resolution) image, crop dark edges and resize
def read_and_resize(filename, out_size, size_is_min, intermediate_size=1024):
    img = cv2.imread(filename)

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(gray,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours)

    if w > 200 and h > 200:
        img = img[y:y + h, x:x + w]
        height, width, _ = img.shape

        if max([height, width]) > intermediate_size:
            ratio = float(intermediate_size / max([height, width]))
            img = cv2.resize(img,
                             tuple([int(width * ratio), int(height * ratio)]),
                             interpolation=cv2.INTER_CUBIC)

    img = resize(img, out_size, size_is_min)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(file, image_size=None, size_is_min=True):
    if _JPEG4PY:
        image = jpeg4py.JPEG(file).decode()
    else:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = resize(image, image_size, size_is_min)
    return image


class ImageDataset(Dataset):
    def __init__(self,
                 files,
                 labels=None,
                 transform=None,
                 buffer_size=20000,
                 image_size=None,
                 size_is_min=True):
        """

        :param files: list of image paths
        :param labels: list of labels
        :param transform: albumentations image transform function
        :param buffer_size: number of images to keep in memory (to reduce disk reads)
        :param image_size: resize images to this size before passing them to transform function
        """
        if labels is not None:
            assert len(files) == len(labels)

        self.files = files
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.size_is_min = size_is_min
        self.buffer_size = buffer_size
        self.load_buffer()

    def load_buffer(self):
        if len(self.files) > self.buffer_size:
            files = self.files[:self.buffer_size]
        else:
            files = self.files
        self.buffer = Parallel(backend='multiprocessing', n_jobs=cpu_count(),
                               verbose=False)(delayed(load_image)(file,
                                                                  self.image_size,
                                                                  self.size_is_min) for file in files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if index >= self.buffer_size:
            image = load_image(self.files[index], self.image_size, self.size_is_min)
        else:
            image = self.buffer[index]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image


class UnlabeledImageDataset(Dataset):
    def __init__(self,
                 files,
                 transforms=None,
                 image_size=None,
                 size_is_min=True):
        self.files = files
        self.transforms = transforms
        self.image_size = image_size
        self.size_is_min = size_is_min

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = read_and_resize(self.files[index], self.image_size, self.size_is_min)

        if self.transforms is not None:
            return [transform(image=image)["image"] for transform in self.transforms]
        else:
            return [image, ]
