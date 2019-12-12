#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader

from lib.dataset import UnlabeledImageDataset
from lib.efficientnet import EfficientNet


# In[4]:


def eval_model(model, dataloader, device, state_lst):
    model.eval()
    model_out_lst = [list() for _ in state_lst]
    with torch.no_grad():
        for xb_lst in dataloader:
            xb_lst = [x.to(device) for x in xb_lst]
            for i, state in enumerate(state_lst):
                model.load_state_dict(state)
                outs = []
                for xb in xb_lst:
                    out = model(xb)
                    if out.shape[1] == 1:
                        out = out.squeeze(dim=1)
                    outs.append(out.detach().cpu())
                model_out_lst[i].append(outs)
        res = []
        for i in range(len(model_out_lst)):
            for j in range(len(model_out_lst[i][0])):
                combined = torch.cat([x[j] for x in model_out_lst[i]])
                res.append(combined)
    return res


# In[5]:


def main():
    IMG_SIZE = [280, 280]
    DF_PATH = r"/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
    IMG_PATH = r"/kaggle/input/aptos2019-blindness-detection/test_images"

    # This folder contains pytorch model state dicts to be used in ensemble
    CHECKPOINT_PATH = r"/kaggle/input/efficientnet-baseline-1/*.pth"
    BATCH_SIZE = 120
    N_CLASSES = 1
    efficientnet_b = 1

    normalize = [[0.43823998, 0.29557559, 0.20054542],
                 [0.27235733, 0.19562355, 0.16674458]]

    df_test = pd.read_csv(DF_PATH)
    X = df_test.id_code.values
    X = IMG_PATH + "/" + X + ".png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transforms = [
        albumentations.Compose([
            albumentations.Normalize(*normalize, p=1),
            albumentations.PadIfNeeded(*IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensor(),
        ]),

        albumentations.Compose([
            albumentations.VerticalFlip(p=1),
            albumentations.Normalize(*normalize, p=1),
            albumentations.PadIfNeeded(*IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensor(),
        ]),
    ]

    dataset_valid = UnlabeledImageDataset(
        files=X,
        transforms=transforms,
        image_size=max(*IMG_SIZE),
        size_is_min=False)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=BATCH_SIZE,
                                  num_workers=4,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False, )

    model = EfficientNet(b=efficientnet_b,
                         in_channels=3,
                         in_spatial_shape=IMG_SIZE,
                         n_classes=N_CLASSES,
                         activation=nn.LeakyReLU(0.001),
                         bias=False,
                         drop_connect_rate=0.2,
                         dropout_rate=None,
                         bn_epsilon=1e-3,
                         bn_momentum=0.01,
                         pretrained=False,
                         progress=False)
    model.to(device)

    state_lst = [torch.load(p) for p in glob.glob(CHECKPOINT_PATH)]
    y_pred_lst = eval_model(model,
                            dataloader_valid,
                            device,
                            state_lst)

    #     print(len(y_pred_lst), type(y_pred_lst[0]), y_pred_lst[0].shape)

    y_pred_median, _ = torch.median(torch.stack(y_pred_lst), dim=0)

    #     print(y_pred_median.shape)
    y_pred_median = torch.round(y_pred_median)
    y_pred_median = torch.where(y_pred_median >= 4.5, torch.tensor([4.0]), y_pred_median)
    y_pred_median = torch.where(y_pred_median < 0.0, torch.tensor([0.0]), y_pred_median)
    y_pred_median = y_pred_median.numpy().astype(np.int32)

    df_test.diagnosis = y_pred_median
    df_test.to_csv("submission.csv", index=False)


# In[6]:

if __name__ == "__main__":
    main()
