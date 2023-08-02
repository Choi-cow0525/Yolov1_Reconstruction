from utils.configwrapper import ConfigWrapper
from model import Yolov1
from model1 import Yolov12
from data_loader import VOC_Custom_Dataset
from utils.summary import draw_tensorboard
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from loss import yolo_loss
from torchsummary import summary
from torchvision.transforms.functional import to_pil_image

import cv2
import pdb
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import copy
from PIL import Image

def find_index_of_1(lst):
    try:
        index = lst.index(1)
        return index
    except ValueError:
        return -1  # If no element with value 1 is found, return -1

if __name__ == '__main__':

    conf = {
        "data" : {
            "train_path" : "./data/VOCdevkit/2007_train.txt",
            "valid_path" : "./data/VOCdevkit/2007_val.txt",
            "label_path" : "C:/Users/sungj/Documents/Co-op/SIITLAB/YOLO/data/VOCdevkit/VOC2007/labels/",
            "train_img_path" : "./data/VOCdevkit/VOC2007/JPEGImages/",
            "valid_img_path" : "./data/VOCdevkit/VOC2007/JPEGImages/",
            "annotation_path" : "./data/VOCdevkit/VOC2007/labels/",
        },
        "train" : {
            "gpu" : 0,
            "init_lr" : 0.001,
            "mid_lr" : 0.01,
            "fin_lr" : 0.0001,
            "path_runs" : "./runs",
            "batch_size" : 1,
            "use_scheduler" : False,
            "num_workers" : 4,
            "epoch" : 135,
        },
        "conv1" : {
            "in_ch" : 3,
            "out_ch" : 64,
            "ks" : 7,
            "strd" : 2,
        },
        "conv2" : {
            "in_ch" : 64,
            "out_ch" : 192,
            "ks" : 3,
            "strd" : 1,
        },
        "conv3" : {
            "in_ch" : 192,
            "int1_ch" : 128,
            "int2_ch" : 256,
            "out_ch" : 512,
            "ks" : 3,
            "strd" : 1,
        },
        "conv4" : {
            "in_ch" : 512,
            "int1_ch" : 256,
            "int2_ch" : 512,
            "out_ch" : 1024,
            "ks" : 3,
            "strd" : 1,
        },
        "conv5" : {
            "in_ch" : 1024,
            "int_ch" : 512,
            "out_ch" : 1024,
            "ks" : 3,
            "strd" : 1,
        },
        "linear" : {
            "in_dim" : 7*7*1024,
            "int_dim" : 496, # 4096
            "out_dim" : 7*7*30,
        },
        "grid" : {
            "S" : 7,
            "B" : 2,
            "C" : 20,
        },
        "gpu" : '0',
    }
    wconf = ConfigWrapper(**conf)

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    ## GPU ##
    os.environ["CUDA_VISIBLE_DEVICES"] = wconf.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device for train : {device}")
    nb_device = torch.cuda.device_count()

    model = Yolov1(wconf)
    model.load_state_dict(torch.load("./checkpoint/runscheckpoint1.pt")['model_state_dict'])
    model.to(device)
    model.eval()

    image1 = Image.open("./image/000020.jpg")
    image3 = Image.open("./image/1.jpg").convert('RGB')
    print(image1.size)
    w = image1.size[0]
    h = image1.size[1]
    image = image1.resize((448, 448))

    with torch.no_grad():
        trans = ToTensor()
        image = trans(image)
        print(image.shape)
        image = image.unsqueeze(0)
        print(image.shape)
        image = image.to(device)
        # .view([1, 3, 448, 448])
        result = model(image)

        print(result.shape)
        cvimage = cv2.imread("./image/000020.jpg")

        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                newresult = result[0][i][j]
                if result[0][i][j][29] > 0.3:
                    cls = result[0][i][j][:20].tolist()
                    print(cls)
                    max_val = max(cls)
                    idx = cls.index(max_val)
                    name = classes[idx]
                    print(name)
                    cx, cy = (j + newresult[25]) * w / 7, (i + newresult[26]) * h / 7
                    iw, ih = w * newresult[27], h * newresult[28]
                    x_min, y_min = cx - iw / 2, cy - ih / 2
                    x_max, y_max = cx + iw / 2, cy + ih / 2
                    color = (0, 255, 0)  # Green color for the box (BGR format)
                    thickness = 2  # Line thickness for the box
                    cv2.rectangle(cvimage, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
                    
        cv2.imshow("Image with Box", cvimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
