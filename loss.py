import torch
import numpy as np

def yolo_loss(loss, nb_device):
    # nb_device 개수가 여러 개이면, squeeze를 해서 n x 1을 n으로 만들기
    
    return