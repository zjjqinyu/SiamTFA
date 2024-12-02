import os
import ltr.admin.settings as ws_settings
import torch

def get_tracker_settings(settings=None):
    if settings is None:
        settings = ws_settings.Settings()
    settings.description = 'Default train settings.'

    settings.multi_gpu = False
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    settings.device = torch.device("cuda:0")

    settings.batch_size = 6
    settings.num_workers = 8
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]

    settings.template_area_factor = 2.0
    settings.template_sz = 224
    settings.search_area_factor = 5.0
    settings.search_sz = 384

    settings.center_jitter_factor = {'template': 0, 'search': 4.0}
    settings.scale_jitter_factor = {'template': 0, 'search': 0.5}

    settings.head_type = 'CENTER'

    settings.backbone_down_sampling = 32

    settings.template_feat_sz = settings.template_sz//settings.backbone_down_sampling
    settings.search_feat_sz = settings.search_sz//settings.backbone_down_sampling
    
    settings.fusion_network_output_channels = 512

    settings.head_channels = 256
    
    return settings