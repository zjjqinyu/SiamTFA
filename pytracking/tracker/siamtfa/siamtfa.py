from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import torchvision.transforms.functional as tvisf
import math
from ltr.data.processing_utils import sample_target
from util.box_ops import clip_box, box_xyxy_to_xywh
from util.hann import hann2d

class SiamTFA(BaseTracker):
    def __init__(self, params):
        super(SiamTFA, self).__init__(params)
        network = params.net
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        if not self.params.has('device'):
            self.params.device = torch.device('cuda') if self.params.use_gpu else torch.device('cpu')
        self.network = network.to(self.params.device)
        self.network.eval()
        self.state = None

        self.feat_sz = params.settings.search_sz // params.settings.backbone_down_sampling
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.params.device) ** 3

    def _img_process(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        img = tvisf.to_tensor(img)
        img = tvisf.normalize(img, mean, std, False).unsqueeze(0)
        img = img.to(self.params.device)
        return img

    def initialize(self, img_rgb, img_aux, info: dict):
        # forward the template once
        template_rgb_crop, _ = sample_target(img_rgb, np.array(info['init_bbox']), 
                                         self.params.settings.template_area_factor,
                                         self.params.settings.template_sz)
        template_aux_crop, _ = sample_target(img_aux, np.array(info['init_bbox']), 
                                         self.params.settings.template_area_factor,
                                         self.params.settings.template_sz)
        template_rgb_crop = self._img_process(template_rgb_crop)
        template_aux_crop = self._img_process(template_aux_crop)
        with torch.no_grad():
            self.network.template(template_rgb_crop, template_aux_crop)

        # save states
        self.state = info['init_bbox']  # (x,y,w,h)
        self.frame_id = 0

    def track(self, img_rgb, img_aux, info: dict = None):
        self.frame_id += 1
        search_rgb_crop, resize_factor = sample_target(img_rgb, np.array(self.state), 
                                                       self.params.settings.search_area_factor,
                                                       self.params.settings.search_sz)
        search_aux_crop, _ = sample_target(img_aux, np.array(self.state), 
                                                       self.params.settings.search_area_factor,
                                                       self.params.settings.search_sz)
        search_rgb_crop = self._img_process(search_rgb_crop)
        search_aux_crop = self._img_process(search_aux_crop)

        with torch.no_grad():
            out_dict = self.network.track(search_rgb_crop, search_aux_crop)

        if self.network.head_type == 'CENTER':
            pred_score_map = out_dict['score_map']
            response = self.output_window * pred_score_map
            pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])   # norm(cx, cy, w, h)
            pred_boxes = pred_boxes.view(-1, 4)
        elif self.network.head_type == 'CORNER':
            pred_boxes = out_dict['pred_boxes']    # norm(cx, cy, w, h)
        else:
            raise NotImplementedError
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.settings.search_sz / resize_factor).tolist()  # (cx, cy, w, h) 
        # Convert to size relative to the original image
        H, W, _ = img_rgb.shape
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)  # (x, y, w, h)
        # state = self.map_box_back(pred_box*self.params.settings.search_sz/resize_factor, resize_factor, self.params.settings.search_sz)
        # H, W, _ = img_rgb.shape
        # self.state = clip_box(state, H, W, margin=10)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.settings.search_sz / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def get_tracker_class():
    return SiamTFA
