import torch.nn as nn
from util import box_ops

class IouLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, gts):
        '''forward function
            input format must be (x, y, h, w)
        Args:
            preds (tensor): The shape should be (n, 4).
            gts (tensor): The shape should be the same as preds.

        Returns:
            1-giou, iou: giou loss and iou.
        '''
        giou, iou = box_ops.generalized_box_iou(preds, gts)
        giou = giou.diag().mean()
        iou = iou.diag().mean()
        return 1-giou, iou
        

        