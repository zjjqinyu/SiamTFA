from . import BaseActor
from util.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from util.heap_map import generate_heatmap
import torch

class MyActor(BaseActor):
    """Actor for training network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'cls': 1.0, 'reg_bbox': 1.0, 'reg_iou': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        pred = self.net(data)
        loss, stats = self.compute_loss(pred, data)
        return loss, stats
    
    def compute_loss(self, pred, data):
        if self.net.head_type == 'CENTER':
            # gt gaussian map
            gt_gaussian_maps = generate_heatmap(data['search_rgb_anno'].unsqueeze(0), pred['score_map'].shape[-1])
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        elif self.net.head_type == 'CORNER':
            pass
        else:
            raise NotImplementedError
     # Get boxes
        pred_boxes = pred['pred_boxes'].unsqueeze(1)   # (B,1,4), norm(x,y,w,h)
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4), norm(x1,y1,x2,y2)
        gt_bbox = data['search_rgb_anno']  # (batch, 4), norm(x, y, w, h)
        num_queries = pred_boxes.size(1)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (BN,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except Exception as e:
            print(e)
            giou_loss, iou = torch.tensor(0.0).to(pred_boxes_vec.device), torch.tensor(0.0).to(pred_boxes_vec.device)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred:
            location_loss = self.objective['focal'](pred['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        total_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # status for log
        mean_iou = iou.detach().mean()
        stats = {"Loss/total": total_loss.item(),
                 "Loss/giou": giou_loss.item(),
                 "Loss/l1": l1_loss.item(),
                 "Loss/location": location_loss.item(),
                 "IoU": mean_iou.item()}
        return total_loss, stats