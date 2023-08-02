import torch
from torch.utils.tensorboard import SummaryWriter

# https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html
def draw_tensorboard(path, losses, set_name, epoch, etc=None):
    writer = SummaryWriter(path)

    ## Main Losses ##
    writer.add_scalar(f"{set_name}/0-Total", losses["loss_total"], epoch)
    # writer.add_scalar(f"{set_name}/1-Box", losses["loss_boxes"], epoch)
    # writer.add_scalar(f"{set_name}/2-Class", losses["loss_class"], epoch)

    writer.flush()
    writer.close()