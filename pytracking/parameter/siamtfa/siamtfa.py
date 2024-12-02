from ltr.train_settings.siamtfa.siamtfa_tracker_settings import get_tracker_settings
from ltr.models.tracking.siamtfa import siamtfa_tracker
from pytracking.utils import TrackerParams
import torch

def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.checkpoint = '/home/qinyu/project/SiamTFA/checkpoints/ltr/siamtfa/siamtfa_tracker/SiamTFA_ep0120.pth.tar'
    params.settings = get_tracker_settings()
    params.device = torch.device("cuda:0")
    params.net = siamtfa_tracker(params.settings)

    return params
