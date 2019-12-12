import argparse
import os
import random

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler

from lib.dataset import ImageDataset
# EfficientNet implementation from https://github.com/abhuse/pytorch-efficientnet
from lib.efficientnet import EfficientNet
from lib.experiment import Experiment


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pytorch_f1(y_pred, y_true, **kwargs):
    if y_pred.ndim == 2:
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.round()
        y_pred = np.where(y_pred > 4.5, 4.0, y_pred)
        y_pred = np.where(y_pred < 0.0, 0.0, y_pred)
        y_pred = y_pred.astype(np.int32)
    y_true = y_true.cpu().numpy()
    return f1_score(y_pred, y_true, **kwargs)


def qw_kappa(y_pred, y_true):
    if y_pred.ndim == 2:
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.round()
        y_pred = np.where(y_pred > 4.5, 4.0, y_pred)
        y_pred = np.where(y_pred < 0.0, 0.0, y_pred)
        y_pred = y_pred.astype(np.int32)
    y_true = y_true.cpu().numpy()
    return cohen_kappa_score(y_pred, y_true, weights='quadratic')


def calc_sampler_weights(y, weights):
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    assert len(unique) == len(weights)
    y_weights = np.zeros_like(y, dtype=np.float32)
    for i, weight in zip(unique, weights):
        y_weights[y == i] = weight
    y_weights /= np.sum(y_weights)
    return y_weights


# In[ ]:
def main(args):
    sid = args.sid
    RND_STATE = 1234
    BATCH_SIZE = 48
    IMG_SIZE = 280
    n_classes = 1
    learning_rate = 2e-4

    efficientnet_b = 1
    cv_folds = 5

    seed_everything(RND_STATE + sid)

    IMG_PATH_2019_TRAIN = r"input/2019_train"
    DF_PATH_2019_TRAIN = r"input/trainLabels19_unique.csv"

    IMG_PATH_2015_TRAIN = r"input/2015_train"
    DF_PATH_2015_TRAIN = r"input/trainLabels15.csv"

    IMG_PATH_2015_TEST = r"input/2015_test"
    DF_PATH_2015_TEST = r"input/testLabels15.csv"

    IMG_PATH_MESSIDOR = r"input/messidor1_jpg"
    DF_PATH_MESSIDOR = r"input/messidor1_labels_adjudicated.csv"

    df_train = pd.read_csv(DF_PATH_2019_TRAIN)
    X_2019_train = df_train.id_code.values
    X_2019_train = IMG_PATH_2019_TRAIN + "/" + X_2019_train + ".jpg"
    y_2019_train = df_train.diagnosis.values.astype(np.float32)

    df_train_2015_train = pd.read_csv(DF_PATH_2015_TRAIN)
    X_2015_train = df_train_2015_train.image.values
    X_2015_train = IMG_PATH_2015_TRAIN + "/" + X_2015_train + ".jpg"
    y_2015_train = df_train_2015_train.level.values.astype(np.float32)

    df_train_2015_test = pd.read_csv(DF_PATH_2015_TEST)

    X_2015_test = df_train_2015_test.image.values
    X_2015_test = IMG_PATH_2015_TEST + "/" + X_2015_test + ".jpg"
    y_2015_test = df_train_2015_test.level.values.astype(np.float32)

    df_messidor = pd.read_csv(DF_PATH_MESSIDOR)
    df_messidor = df_messidor[df_messidor.adjudicated_dr_grade > -1]
    X_messidor = df_messidor.image.values
    X_messidor = IMG_PATH_MESSIDOR + "/" + X_messidor + ".jpg"
    y_messidor = df_messidor.adjudicated_dr_grade.values.astype(np.float32)

    normalize = [[0.43823998, 0.29557559, 0.20054542],
                 [0.27235733, 0.19562355, 0.16674458]]

    img_size = (IMG_SIZE, IMG_SIZE)
    transform_train = albumentations.Compose([
        albumentations.RandomCrop(*img_size),
        albumentations.HueSaturationValue(hue_shift_limit=7),
        albumentations.RandomBrightnessContrast(),
        albumentations.ShiftScaleRotate(shift_limit=0,
                                        scale_limit=(-0.05, 0.15),
                                        interpolation=cv2.INTER_CUBIC),
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
        albumentations.Blur(),
        albumentations.Normalize(*normalize, p=1),
        ToTensorV2(),
    ])

    transform_validation = albumentations.Compose([
        albumentations.CenterCrop(*img_size),
        albumentations.Normalize(*normalize, p=1),
        ToTensorV2(),
    ])

    skf9 = StratifiedKFold(n_splits=cv_folds, random_state=RND_STATE, shuffle=True)

    for split_id, (tra9, tes9) in enumerate(skf9.split(X_2019_train, y_2019_train)):

        if split_id != sid:
            continue
        X_aptos_train, X_aptos_valid = X_2019_train[tra9], X_2019_train[tes9]
        y_aptos_train, y_aptos_valid = y_2019_train[tra9], y_2019_train[tes9]

        X_train = np.concatenate([
            X_aptos_train,
            X_messidor,
            X_2015_train,
            X_2015_test,
        ])
        y_train = np.concatenate([
            y_aptos_train,
            y_messidor,
            y_2015_train,
            y_2015_test,
        ])

        X_valid = np.concatenate([
            X_aptos_valid,
        ])
        y_valid = np.concatenate([
            y_aptos_valid,
        ])

        print("train/validation set size: {}/{}".format(len(y_train), len(y_valid)))

        dataset_train = ImageDataset(
            files=X_train,
            labels=y_train,
            transform=transform_train,
            buffer_size=10000,  # lower this value if out-of-memory is thrown <<<<<<<<<<<<<<<<<<<<
            image_size=IMG_SIZE)

        dataset_valid = ImageDataset(
            files=X_valid,
            labels=y_valid,
            transform=transform_validation,
            buffer_size=0,
            image_size=IMG_SIZE,
            size_is_min=True)

        # sampling weight for inputs of each class
        weights = np.array([1, 5, 5, 10, 10])
        weights = calc_sampler_weights(y_train, weights)
        # increase probability of selecting aptos 2019 train images by 5 times
        weights[:y_aptos_train.shape[0]] *= 5

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=BATCH_SIZE,
                                      num_workers=4,
                                      # shuffle=True,
                                      sampler=WeightedRandomSampler(weights, 45000, True),
                                      pin_memory=True,
                                      drop_last=True,
                                      )
        dataloader_valid = DataLoader(dataset_valid,
                                      batch_size=BATCH_SIZE,
                                      num_workers=4,
                                      shuffle=False,
                                      pin_memory=True,
                                      drop_last=False,
                                      )

        _, train_val_ids = train_test_split(list(range(len(X_train))),
                                            test_size=0.1,
                                            stratify=y_train,
                                            random_state=RND_STATE)

        train_val_sampler = SubsetRandomSampler(train_val_ids)
        dataloader_train_eval = DataLoader(dataset_train,
                                           batch_size=BATCH_SIZE,
                                           num_workers=4,
                                           sampler=train_val_sampler,
                                           pin_memory=True,
                                           drop_last=False,
                                           )

        model = EfficientNet(b=efficientnet_b,
                             in_channels=3,
                             in_spatial_shape=IMG_SIZE,
                             n_classes=n_classes,
                             activation=nn.LeakyReLU(0.001),
                             bias=False,
                             drop_connect_rate=0.2,
                             dropout_rate=None,
                             bn_epsilon=1e-3,
                             bn_momentum=0.01,
                             pretrained=True,
                             progress=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=0.1 * learning_rate)
        # optimizer = optim.RMSprop(model.parameters(),
        #                        lr=learning_rate,
        #                        momentum=0.9,
        #                        alpha=0.9,
        #                        weight_decay=0.1 * learning_rate)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.SmoothL1Loss()
        eval_metrics = [("loss", criterion, {}),
                        ("f1_score", pytorch_f1, {"average": "macro"}),
                        # ("classwise_f1", pytorch_f1, {"average": None}),
                        ("qwk", qw_kappa, {})]

        scheduler = None

        s = ("{epoch}:{step}/{max_epoch} | {loss_train:.4f} / {loss_valid:.4f}"
             " | {f1_score_train:.4f} / {f1_score_valid:.4f}"
             # " | {classwise_f1_train}/{classwise_f1_valid}"
             " | {qwk_train:.4f} / {qwk_valid:.4f} | {time_delta}")
        exp = Experiment(dl_train=dataloader_train,
                         dl_train_val=dataloader_train_eval,
                         dl_validation=dataloader_valid,
                         model=model,
                         optimizer=optimizer,
                         criterion=criterion,
                         device=device,
                         max_epoch=20,
                         metrics=eval_metrics,
                         target_metric="qwk",
                         format_str=s,
                         scheduler=scheduler,
                         load_path=None,
                         save_path="save/b%d_%dpx/%d" % (efficientnet_b, IMG_SIZE, split_id),
                         evaluate_freq=3)

        exp.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sid", type=int, required=True, help="Cross-Validation fold id")
    args = parser.parse_args()
    main(args)
