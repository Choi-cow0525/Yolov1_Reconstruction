## Thanks to https://pjreddie.com/ for the code
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd # get current working directory
from os.path import join
from tqdm import tqdm

sets = [('2007', 'train'), ('2007', 'val')] #, ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def normalize(size, box):
    dw = 1./size[0] # return float -> normalize size w to 1
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    y = y*dh
    w = w*dw
    h = h*dh
    return (x, y, w, h)

def convert_xml2txt(year, image_id):
    in_file = open(f'./data/VOCdevkit/VOC{year}/Annotations/{image_id}.xml', 'r') # original annotation as xml
    out_file = open(f'./data/VOCdevkit/VOC{year}/labels/{image_id}.txt', 'w') # new annotation created as txt
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'): # covers various object
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox') # bndbox contains information about bounding boxes(xmin, xmax, ymin, ymax)
        coord = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        norm_coord = normalize((w, h), coord)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in norm_coord]) + '\n')


if __name__ == "__main__":
    wd = getcwd()
    wd = wd.replace("\\", "/")
    print(wd)

    for year, image_set in sets:
        if not os.path.exists(f"./data/VOCdevkit/VOC{year}/labels/"):
            os.makedirs(f'./data/VOCdevkit/VOC{year}/labels/')
        image_ids = open(f"./data/VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt").read().strip().split()
        list_file = open(f"./data/VOCdevkit/{year}_{image_set}.txt", 'w')
        for image_id in tqdm(image_ids):
            list_file.write(f"{wd}/data/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg\n")
            convert_xml2txt(year, image_id)
        list_file.close()
