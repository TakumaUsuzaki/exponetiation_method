import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import cv2
import glob
from statistics import mean, median, variance, stdev
import numpy as np
import pandas as pd
from PIL import Image
import json
import random
import math
import os
import tqdm


class MyDataset(Dataset):

    def __init__(self, pathes, exponent=1, transform=False):
        self.pathes = pathes
        self.exponent = exponent
        self.data_im = []
        self.label = []
        self.transform = transform

        for path in self.pathes:
            basename = os.path.basename(path)
            im = Image.open(path)
            im = np.asarray(im, dtype='float64')
            im = im.transpose(2, 0, 1)
            img = im.copy()
            img.flags['WRITEABLE'] = True

            if self.transform:
                img = torch.tensor(img, dtype=torch.float)
                max_p = torch.max(img)
                img_t = img**self.exponent/(max_p**self.exponent)
                img = img/255.0

                img = torch.cat([img, img_t], axis=1)
                self.data_im.append(img)
                # self.data_im.append(img_t)

                if "mal1" in basename:
                    self.label.append(0)
                    # self.label.append(0)

                elif "mal5" in basename:
                    self.label.append(1)
                    # self.label.append(1)

            if not self.transform:
                img = torch.tensor(img, dtype=torch.float)
                img = img/255.0
                self.data_im.append(img)

                if "mal1" in basename:
                    self.label.append(0)

                elif "mal{}".format(mal) in basename:
                    self.label.append(1)

    def __len__(self):
        return (len(self.label))

    def __getitem__(self, idx):
        out_im = self.data_im[idx]
        out_label = self.label[idx]

        return (out_im, out_label)
