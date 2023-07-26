import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ObjectInBox:
    def __init__(self, conf, ground_truth) -> None:
        self.conf = conf
        self.gt = ground_truth
        self.ifObjLabel = torch.zeros(7, 7)
        self.gt_boxes = []

        # My Mistake - misunderstood the "resposibility part" of the paper
        # no need to concern other boxes who do not contain object == ** the center of the object is not in that box
        # for batch in range(conf.train.batch_size):
        #     for i in range(conf.grid.S):
        #         for j in range(conf.grid.S):
        #             if self.gt[batch][i][j][23] != 0 & self.gt[batch][i][j][24] != 0: # if w & h is not zero
        #                 newbox = self.gt[batch][i][j][20:] # x, y, w, h, c
        #                 newbox[0] = newbox[0] + j
        #                 newbox[1] = newbox[1] + i
        #                 newbox[2] = newbox[2] * 7
        #                 newbox[3] = newbox[3] * 7
        #                 self.gt_boxes.append(newbox)
            
        #     for box in self.gt_boxes:
        #         lefthigh = (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2))
        #         rightlow = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))
        #         for i in range(rightlow[0] - lefthigh[0] + 1):
        #             for j in range(rightlow[1] - lefthigh[1] + 1):
        #                 self.ifObjLabel[j+lefthigh[1]][i+lefthigh[0]] = 1
        # print(self.ifObjLabel)
        print("Done Init\n")

    # return True if IOU is highest when object exist in that cell / return False if no object exist
    def doIOU(self, i, j, boxes) -> bool: # 박스 개수만큼 for loop을 돌리면 됨
        assert list(boxes.shape) == self.conf.grid.C + self.conf.grid.B * 5
        assert list(self.gt.shape) == self.conf.grid.C + 5
        bigger = 0
        gt_box = self.gt[i][j][20:]
        print(f"gt_box : {gt_box}")
        for index in range(list(boxes.shape)[0] // 5):
            one_box = boxes[20 + 5 * index: 20 + 5 * (index+1)]
            print(f"{index}th box is : {one_box}\n")
            one_box

        return True

    # return True if ObjExist if not return False -> substituted by init
    def ifObjExist(self, boxes) -> bool:
        assert list(boxes.shape) == self.conf.grid.C + self.conf.grid.B * 5
        assert list(self.gt.shape) == self.conf.grid.C + 5

        return True

# https://brunch.co.kr/@kmbmjn95/35
def yolo_loss(conf, result, ground_truth) -> float:
    # nb_device 개수가 여러 개이면, squeeze를 해서 n x 1을 n으로 만들기
    # input dim of result is 7 x 7 x 30
    # input dim of ground_truth is 7 x 7 x 25

    lamb_coord = 5
    lamb_no_obj = 0.5
    total_loss = 0
    gt_ifobj = ObjectInBox(conf, ground_truth).ifObjLabel
    # https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation
    for batch in range(result.shape[0]):
        newresult = result[batch]
        print(newresult.shape)
        for i in range(newresult.shape[0]):
            for j in range(newresult.shape[1]):
                tresult = newresult[i][j]
                gt = ground_truth[batch][i][j] # the boxes are already correctly labelled from data_loader.py
                # do loss5 - classification loss
                total_loss += torch.sum(torch.pow((tresult[:20] - gt[:20]), 2))

                if gt_ifobj[i][j] == 0:
                    # do loss4
                    # lamb_no_obj * if_no_obj_exist * (confidence - confidencet)^2
                    for n in range(conf.grid.B):
                        total_loss += lamb_no_obj * torch.pow(tresult[24 + 5*n] - gt[-1], 2)

                else:
                    for n in range(conf.grid.B):
                        newloss = gt_ifobj.doIOU(i, j, tresult)
                        if newloss:
                        # loss1
                        # lamb_coord * if_box_responsible * [(x - xt)^2 + (y-yt)^2]
                            total_loss += lamb_coord * (torch.pow(tresult[20 + 5*n] - gt[20], 2) + torch.pow(tresult[21 + 5*n] - gt[21], 2))
                        # loss2
                        # lamb_coord * if_box_responsible * [(sqrt(w) - sqrt(wt))^2 + (sqrt(h) - sqrt(ht))^2]
                            total_loss += lamb_coord * (torch.pow(torch.sqrt(tresult[22 + 5*n]) - torch.sqrt(gt[22]), 2) + torch.pow(torch.sqrt(tresult[23 + 5*n]) - torch.sqrt(gt[23]), 2))
                        # loss3
                        # if_box_responsible * (confidence - confidencet)^2
                            total_loss += torch.pow(tresult[24 + 5*n] - gt[-1], 2)
                            break
    return total_loss