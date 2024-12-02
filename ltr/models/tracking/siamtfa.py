import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.fusion as fusions
import ltr.models.head as heads
from ltr import model_constructor
from pytracking import TensorDict
from util.box_ops import box_xyxy_to_cxcywh

class SiamTFA(nn.Module):
    """The SiamTFA network.
    args:
        backbone:  Backbone feature extractor network.
        fusion_network:  To fuse template and search features.
        head:  Bounding box prediction network."""

    def __init__(self, backbone, fusion_network, head, head_type):
        super().__init__()
        self.backbone = backbone
        self.fusion_network = fusion_network
        self.head = head
        self.head_type = head_type
        
    def forward(self, data):
        """Runs the network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking."""
        template_fus_feat = self.backbone(data['template_rgb_images'], data['template_aux_images'])
        search_fus_feat = self.backbone(data['search_rgb_images'], data['search_aux_images'])
        fusion_feat = self.fusion_network(template_fus_feat, search_fus_feat)
        output = self._head_forward(fusion_feat)
        return output
    
    def template(self, template_rgb_images, template_aux_images):
        """For tracking."""
        self.template_fus_feat= self.backbone(template_rgb_images, template_aux_images)

    def track(self, search_rgb_images, search_aux_images):
        """For tracking."""
        search_fus_feat = self.backbone(search_rgb_images, search_aux_images)
        fusion_feat = self.fusion_network(self.template_fus_feat, search_fus_feat)
        output = self._head_forward(fusion_feat)
        return output
    
    def _head_forward(self, fusion_feat):
        output = {}
        if self.head_type == 'CENTER':
            score_map, pred_boxes, size_map, offset_map  = self.head(fusion_feat)
            output['score_map'] = score_map
            output['pred_boxes'] = pred_boxes     # (cx, cy, w, h)
            output['size_map'] = size_map
            output['offset_map'] = offset_map
        elif self.head_type == 'CORNER':
            pred_boxes = self.head(fusion_feat)    # (x1, y1, x2, y2)
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)   # (cx, cy, w, h)
            output['pred_boxes'] = pred_boxes
        else:
            raise NotImplementedError
        return TensorDict(output)

@model_constructor
def siamtfa_tracker(settings):
    # Backbone
    backbone = backbones.tfa_swin_backbone()
    fusion_network = fusions.pixelwise_xcorr(template_feat_sz=settings.template_feat_sz,
                                             search_feat_sz=settings.search_feat_sz,
                                             out_channels=settings.fusion_network_output_channels)
    if settings.head_type == 'CENTER':
        head = heads.center_head(inplanes=settings.fusion_network_output_channels,
                                channel=settings.head_channels,
                                feat_sz=settings.search_feat_sz,
                                stride = settings.backbone_down_sampling)
    elif settings.head_type == 'CORNER':
        head = heads.coner_head(inplanes=settings.fusion_network_output_channels,
                                channel=settings.head_channels,
                                feat_sz=settings.search_feat_sz,
                                stride = settings.backbone_down_sampling)
    else:
        raise NotImplementedError
    net = SiamTFA(backbone, fusion_network, head, settings.head_type)
    return net