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
        assert list(boxes.shape)[0] == self.conf.grid.C + self.conf.grid.B * 5
        assert list(self.gt.shape)[2] == self.conf.grid.C + 5

        best = 0
        bigger = 0
        gt_box = self.gt[i][j][20:]
        # print(f"input box is : {boxes}\n")
        # print(f"gt_box : {gt_box}\n")
        x, y, w, h = gt_box[0], gt_box[1], gt_box[2], gt_box[3]
        gt_lefthigh = (j + x - self.conf.grid.S * w / 2, i + y - self.conf.grid.S * h / 2)
        gt_rightlow = (j + x + self.conf.grid.S * w / 2, i + y + self.conf.grid.S * h / 2)

        for index in range(self.conf.grid.B):
            one_box = boxes[20 + 5 * index: 20 + 5 * (index+1)]
            # print(f"{index}th box is : {one_box}\n")
            x, y, w, h = one_box[0], one_box[1], one_box[2], one_box[3]
            lefthigh = (j + x - self.conf.grid.S * w / 2, i + y - self.conf.grid.S * h / 2)
            rightlow = (j + x + self.conf.grid.S * w / 2, i + y + self.conf.grid.S * h / 2)
            # print(f"lefth : {lefthigh}")
            # print(f"rightl : {rightlow}")

            x1 = torch.max(gt_lefthigh[0], lefthigh[0])
            y1 = torch.max(gt_lefthigh[1], lefthigh[1])
            x2 = torch.max(gt_rightlow[0], rightlow[0])
            y2 = torch.max(gt_rightlow[0], rightlow[0])

            intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
            box1_area = abs((lefthigh[0] - rightlow[0]) * (lefthigh[1] - rightlow[1]))
            box2_area = abs((gt_lefthigh[0] - gt_rightlow[0]) * (gt_lefthigh[1] - gt_rightlow[1]))

            curr = intersection / (box1_area + box2_area - intersection + 1e-6) # prevent divide by 0
            if curr > best:
                bigger = index

        return bigger

    # return True if ObjExist if not return False -> substituted by init
    def ifObjExist(self, i, j) -> bool:
        assert list(self.gt.shape)[2] == self.conf.grid.C + 5
        if self.gt[i][j][-1] == 1: # if confidence of ground truth is 1
            return True
        else:
            return False


# https://brunch.co.kr/@kmbmjn95/35
def yolo_loss(conf, result, ground_truth) -> float:
    # nb_device 개수가 여러 개이면, squeeze를 해서 n x 1을 n으로 만들기
    # input dim of result is 7 x 7 x 30
    # input dim of ground_truth is 7 x 7 x 25

    lamb_coord = 5
    lamb_no_obj = 0.5
    total_loss = 0
    # print(f"result shape is {result.shape}\n")
    # https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation
    for batch in range(result.shape[0]):
        # print(result)
        # print(f"groud_truth is {ground_truth}")
        gt_ifobj = ObjectInBox(conf, ground_truth[batch]) # the boxes are already correctly labelled from data_loader.py
        newresult = result[batch]
        # print(f"newresult shape is {newresult.shape}\n")
        for i in range(newresult.shape[0]):
            for j in range(newresult.shape[1]):
                tresult = newresult[i][j]
                gt = ground_truth[batch][i][j]
                # print(f"tresult is {tresult}\n")
                # print(f"gt is {gt}\n")

                if not gt_ifobj.ifObjExist(i, j):
                    for n in range(conf.grid.B):
                    # do loss4 - no obj confidence
                        total_loss += lamb_no_obj * (tresult[24 + 5*n] - gt[-1]) ** 2
                        # print(f"loss4 : {total_loss}")

                else:
                    best_iou_idx = gt_ifobj.doIOU(i, j, tresult)
                    for n in range(conf.grid.B):
                        # do loss5 - classification loss
                        # print(tresult[:20] - gt[:20])
                        total_loss += torch.sum((tresult[:20] - gt[:20]) ** 2)
                        # print(f"loss5 : {total_loss}")
                        if n == best_iou_idx:
                        # loss1 - center loss
                            # print(tresult[20 + 5*n] - gt[20])
                            # print(tresult[21 + 5*n] - gt[21])
                            total_loss += lamb_coord * ((tresult[20 + 5*n] - gt[20]) ** 2 + (tresult[21 + 5*n] - gt[21]) ** 2)
                            # print(f"loss1 : {total_loss}")
                        # loss2 - w, h loss
                            total_loss += lamb_coord * ( (torch.sqrt(tresult[22 + 5*n]) - torch.sqrt(gt[22])) ** 2 + (torch.sqrt(tresult[23 + 5*n]) - torch.sqrt(gt[23])) ** 2 )
                            # print(f"loss2 : {total_loss}")
                        # loss3 - confidence loss
                            total_loss += (tresult[24 + 5*n] - gt[-1]) ** 2
                            # print(f"loss3 : {total_loss}")
                            break
                
                #print(f"total_loss is : {total_loss}\n")
    return {"loss_total" : total_loss}