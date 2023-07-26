from utils.configwrapper import ConfigWrapper
from model import Yolov1
from data_loader import VOC_Custom_Dataset
from utils.summary import draw_tensorboard
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from loss import yolo_loss

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import copy

def train(conf, epoch, train_losses):
    model.train()
    for batch_idx, (image, label) in enumerate(dloader_train):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        result = model(image)
        losses = yolo_loss(conf, result, label)

        # https://hongl.tistory.com/363
        loss = losses['loss_total'].clone()
        loss.backward()
        optimizer.step()

        if conf.train.use_scheduler:
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
        elif epoch == 0:
            lr = lr + (conf.train.mid_lr - conf.train.init_lr) / (dataset_train.__len__()//batch_size)
        elif epoch == 74:
            lr = conf.train.init_lr
        elif epoch == 104:
            lr = conf.train.fin_lr
        
        ## print during training ##
        if batch_idx % 20 == 0:
            print(f"<Batch: {batch_idx}>\n lr: {lr}, loss_tot: {losses['loss_total']}, loss_boxes: {losses['loss_boxes']}, loss_class: {losses['loss_class']}")

        ## batch-step-losses ##
        assert len(losses.keys()) == len(train_losses.keys())
        for key in losses.keys():
            # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
            train_losses[key] += losses[key].detach().cpu().item() # x = torch.tensor([10]), x.item() -> 10

    ## epoch-step-losses ##
    for key in losses.keys():
        train_losses[key] /= (batch_idx + 1)

    draw_tensorboard(
        wconf,
        losses=train_losses,
        set_name='Train',
        epoch=epoch,
        etc=[lr]
    )


def valid(conf, epoch, valid_losses):
    # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
    # model.eval() -> 이걸 키면 dropout layer가 작동 멈춤
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(dloader_valid):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            result = model(image)
            losses = yolo_loss(conf, result, label)

            ## batch-step-losses ##
            assert len(losses.keys()) == len(valid_losses.keys())
            valid_losses[key] += losses[key].detach().cpu().item()

    ## epoch-step-losses ##
    for key in losses.keys():
        valid_losses[key] /= (batch_idx + 1)
    print(f"\n\nvalid_loss = loss_tot: {losses['loss_total']}, loss_boxes: {losses['loss_boxes']}, loss_class: {losses['loss_class']}")

    draw_tensorboard(
        conf,
        losses=valid_losses,
        set_name='Valid',
        epoch=epoch,
    )
    
    ## Best ##
    if valid_losses['loss_total'] < best_losses['loss_total']:
        best_epoch = epoch
        best_losses = valid_losses
        os.makedirs(path_params, exist_ok=True)

        if nb_device > 1:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.module.state_dict(),
            }, path_params+f'checkpoint{best_epoch}.pt')

        else:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
            }, path_params+f'checkpoint{best_epoch}.pt')

        print(f"\n\nbest_loss = loss_tot: {losses['loss_total']}, loss_boxes: {losses['loss_boxes']}, loss_class: {losses['loss_class']}")

        draw_tensorboard(
            conf,
            losses=best_losses,
            set_name='Best',
            epoch=best_epoch,
        )

    elif epoch % 20 == 0:
        if nb_device > 1:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.module.state_dict(),
            }, path_params+f'checkpoint{best_epoch}.pt')

        else:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
            }, path_params+f'checkpoint{best_epoch}.pt')
            
# out_file = open(f'VOCdevkit/VOC{year}/labels/{image_id}.xml', 'w') # new annotation created as txt
# list_file = open(f"{year}_{image_set}.txt", 'w') # contains all image path

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
            "batch_size" : 2,
            "use_scheduler" : False,
            "num_workers" : 4,
            "epoch" : 100,
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
            "int_dim" : 4096,
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

    ## GPU ##
    os.environ["CUDA_VISIBLE_DEVICES"] = wconf.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device for train : {device}")
    nb_device = torch.cuda.device_count()

    ## Params ##
    path_dataset = wconf.data.train_img_path
    path_params = wconf.train.path_runs

    ## Data Loader ##
    start = time.time() # return current time
    dataset_train = VOC_Custom_Dataset(wconf, ToTensor(), None, True)
    dataset_valid = VOC_Custom_Dataset(wconf, ToTensor(), None, False)
    
    end = time.time()
    print(f"DataLoading took {end-start} seconds")

    batch_size = \
        wconf.train.batch_size * nb_device if nb_device != 0 else wconf.train.batch_size
    # https://velog.io/@seokjin1013/PyTorch-numworkers%EC%97%90-%EA%B4%80%ED%95%98%EC%97%AC
    workers = \
        wconf.train.num_workers * nb_device if nb_device != 0 else wconf.train.num_workers
    assert batch_size > 1

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    dloader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    print(dloader_train.__len__())
    ## Model ##
    model = Yolov1(wconf)
    if nb_device > 1:
        model = nn.DataParallel(model)
    model.to(device)

    print(f"Model Param Number : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr = wconf.train.init_lr)
    if wconf.train.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = (dataset_train.__len__()//batch_size),
            T_multi = 1,
            eta_min=0.000001)
    
    ## Training ##
    best_epoch = 0
    best_losses = {'loss_total' : 100}
    for epoch in tqdm(range(wconf.train.epoch)):
        train_losses = {
            'loss_total' : 0,
            'loss_boxes' : 0,
            'loss_class' : 0,
        }
        valid_losses = copy.deepcopy(train_losses)

        train(conf, epoch, train_losses)
        valid(conf, epoch, valid_losses)

    print(f"\n<< BEST >> \nepoch: {best_epoch} \
          \nBest_loss: {best_losses['loss_total']} \
          \nBest_loss_boxes: {best_losses['loss_boxes']} \
          \nBest_loss_class: {best_losses['loss_class']}\n")
