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
import dataset_constructor


def exponentiation_dl(exp, n_epoch, batch_size):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device)
    vgg16.double

    epoch = n_epoch

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'TN': []
    }

    train_1 = input("Please Enter Path of category 1 for train: ")
    train_2 = input("Please Enter Path of category 2 for train: ")
    test_1 = input("Please Enter Path of category 1 for test: ")
    test_2 = input("Please Enter Path of category 2 for test: ")
    out_put = input("Please Enter Output path: ")

    pathes_train_all1 = glob.glob(train_1 + '/*.png')
    pathes_train_all2 = glob.glob(train_2 + '/*.png')
    image_number_train = min(len(pathes_train_all1), len(pathes_train_all2))
    pathes_train1 = random.sample(pathes_train_all1, image_number_train)
    pathes_train2 = random.sample(pathes_train_all2, image_number_train)
    pathes_train = pathes_train1 + pathes_train2

    pathes_test_all1 = glob.glob(test_1 + '/*.png')
    pathes_test_all2 = glob.glob(test_1 + '/*.png')
    image_number_test = min(len(pathes_test_all1), len(pathes_test_all2))
    pathes_test1 = random.sample(pathes_test_all1, image_number_test)
    pathes_test2 = random.sample(pathes_test_all2, image_number_test)
    pathes_test = pathes_test1 + pathes_test2

    train_set = dataset_constructor.MyDataset(
        pathes_train, exponent=exp, transform=True)
    test_set = dataset_constructor.MyDataset(
        pathes_test, exponent=exp, transform=True)

    N_train = len(train_set)
    N_test = len(test_set)

    print("The number of images used for train and test is {0} and {1}, respectively.".format(
        N_train, N_test))

    dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(params=vgg16.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    for e in range(epoch):
        #print("epoch {} is being calculated.".format(e+1))

        train_loss = 0.0
        train_correct = 0
        train_toal = 0

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        """ Training Part"""
        vgg16.train(True)
        for i, data in enumerate(dataloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = vgg16(inputs)
            loss = loss_fn(output, labels)

            if not math.isnan(loss.item()):
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_toal += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_correct/train_toal)

        """ Test Part """
        vgg16.eval()

        with torch.no_grad():
            for data in testloader:

                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                output = vgg16(images)
                loss = loss_fn(output, labels)

                if not math.isnan(loss.item()):
                    test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    TP += (2*predicted - labels == 1).sum().item()  # TP
                    FP += (2*predicted - labels == 2).sum().item()  # FP
                    FN += (2*predicted - labels == -1).sum().item()  # FN
                    TN += (2*predicted - labels == 0).sum().item()  # TN

        print("test loss = {0}, test_acc = {1}".format(
            test_loss/test_total, test_correct/test_total))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_correct/N_test)
        history['TP'].append(TP)
        history['FP'].append(FP)
        history['FN'].append(FN)
        history['TN'].append(TN)

        with open(out_put + "/exp{}.json".format(int(exp)), 'w') as outfile:
            json.dump(history, outfile, indent=4)

    torch.cuda.empty_cache()

    return ()
