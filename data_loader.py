import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.configwrapper import ConfigWrapper
from torchvision.transforms import ToTensor

class VOC_Custom_Dataset(Dataset):
    def __init__(self, config, transform, target_transform, istrain) -> None:
        # super().__init__() 
        # # torch.utils.data.Dataset init has no meaningful action
        # https://discuss.pytorch.org/t/dataset-inheritance-does-not-require-super/92945
        self.config = config
        if istrain:
            self.image_path = config.data.train_path
            self.annotation = config.data.annotation_path
        else: # valid
            self.image_path = config.data.valid_path
            self.annotation = config.data.annotation_path
        self.transform = transform
        self.target_transform = target_transform
        with open(self.image_path, 'r') as f:
            self.imgindex = f.readlines()
            # https://stackoverflow.com/questions/10201008/using-readlines-twice-in-a-row
            # self.lines = len(f.readlines())
            # print(f"self.lines is {self.imgindex}")
        
        self.idlist = []
        for line in self.imgindex:
            self.idlist.append(line)
        # print(f"idlist is {self.idlist}")
        # self.idlist becomes below
        # [ C:\Users\sungj\Documents\Co-op\SIITLAB\YOLO/data/VOCdevkit/VOC2007/JPEGImages/000005.jpg,
        #   C:\Users\sungj\Documents\Co-op\SIITLAB\YOLO/data/VOCdevkit/VOC2007/JPEGImages/000007.jpg ]

        self.S = int(self.config.grid.S)
        self.C = int(self.config.grid.C)
        self.B = int(self.config.grid.B)

    def __len__(self):
        return len(self.imgindex)

    def __getitem__(self, index) -> torch.Tensor:
        image = Image.open(self.idlist[index].strip())
        print(image)
        image = image.resize((448, 448))
        label = self.idlist[index][-11:-5]
        annot_path = self.config.data.label_path + label + ".txt"
        
        temp = np.zeros([self.S, self.S, self.C + 5], dtype=float) 
        # model output은 7x7x30 but 어차피 B개의 box 중 하나만 정답으로 사용하므로 label은 20 + 5
        with open(annot_path, 'r') as f:
            line = f.readlines()
        for lin in line:
            wordlist = lin.split(" ")
            cls, xmin, xmax, ymin, ymax = int(wordlist[0]), float(wordlist[1]), float(wordlist[2]), float(wordlist[3]), float(wordlist[4])
            # find position as grid
            # The (x, y) coordinates represent the center of box relative to the bounds of grid cell
            # The width and height are predicted relative to whole image
            w, h = xmax - xmin, ymax - ymin
            cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
            cx, cy = cx * self.S, cy * self.S
            gx, gy = int(cx), int(cy) # grid
            nx, ny = cx - gx * 7, cy - gy * 7 # normalized
        temp[gy][gx][20:25] = [nx, ny, w, h, 1] # make conf to 1
        temp[gy][gx][cls - 1] = 1 # make cls number 1

        target = torch.tensor(temp) # copy numpy to tensor
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
    
    
if __name__ == "__main__":
    conf = {
        "data" : {
            "train_path" : "./data/VOCdevkit/2007_train.txt",
            "valid_path" : "./data/VOCdevkit/2007_valid.txt",
            "label_path" : "C:/Users/sungj/Documents/Co-op/SIITLAB/YOLO/data/VOCdevkit/VOC2007/labels/",
            # "train_img_path" : "./data/VOCdevkit/VOC2007/JPEGImages",
            # "valid_img_path" : "./data/VOCdevkit/VOC2007/JPEGImages",
            "annotation_path" : "./data/VOCdevkit/VOC2007/labels/",
        },
        "grid" : {
            "S" : 7,
            "B" : 2,
            "C" : 20,
        },
    }
    wconf = ConfigWrapper(**conf)
    dataset = VOC_Custom_Dataset(wconf, ToTensor(), None, True)
    first = dataset[0]
    print(first[0])
    print(first[1][0])
    print(first[1][1])
    print(first[1][2])
    print(first[1][3])
    print(first[1][4])
    print(first[1][5])
    print(first[1][6])

# clear!!