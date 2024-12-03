import torch
import torch.nn as nn
import torch.optim as optim
# from ltr.dataset import GTOT, RGBT234, LasHeR, VTUAV
from ltr.dataset.rgbt_datasets import GTOT, RGBT234, LasHeR, VTUAV
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import siamtfa
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.models.loss.iou_loss import IouLoss
from ltr.models.loss.focal_loss import FocalLoss
from .siamtfa_tracker_settings import get_tracker_settings

def run(settings):
    settings = get_tracker_settings(settings)

    # Train datasets
    lasher_train = LasHeR(split='train')
    lasher_test = LasHeR(split='test')

    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip(probability=0.5),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    # transform_train_aux = tfm.Transform(tfm.RandomSpeckleNoise(0.5))
    transform_train_aux = None

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # The tracking pairs processing module
    data_processing_train = processing.MultimodalProcessing(template_area_factor=settings.template_area_factor,
                                                            template_sz=settings.template_sz,
                                                            search_area_factor=settings.search_area_factor,
                                                            search_sz=settings.search_sz,
                                                            center_jitter_factor=settings.center_jitter_factor,
                                                            scale_jitter_factor=settings.scale_jitter_factor,
                                                            mode='sequence',
                                                            transform=transform_train,
                                                            joint_transform=transform_joint,
                                                            aux_transform=transform_train_aux)

    data_processing_val = processing.MultimodalProcessing(template_area_factor=settings.template_area_factor,
                                                          template_sz=settings.template_sz,
                                                          search_area_factor=settings.search_area_factor,
                                                          search_sz=settings.search_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_val,
                                                          joint_transform=transform_joint)

    # Train sampler and loader
    dataset_train = sampler.MultimodalSampler([lasher_train], [1],
                                              samples_per_epoch=2000*settings.batch_size, max_gap=200, 
                                              num_search_frames=1, num_template_frames=1,
                                              processing=data_processing_train)
    loader_train = LTRLoader('train_py1', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True)
    dataset_val = sampler.MultimodalSampler([lasher_test], [1], 
                                            samples_per_epoch=1000*settings.batch_size , max_gap=200,
                                            num_search_frames=1, num_template_frames=1,
                                            processing=data_processing_val)
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                            drop_last=True, epoch_interval=5)

    # Create network and actor
    net = siamtfa.siamtfa_tracker(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        # net = MultiGPU(net, dim=1)
        net = MultiGPU(net)

    objective = {'focal': FocalLoss(), 'l1': nn.L1Loss(), 'giou': IouLoss()}

    loss_weight = {'focal': 1, 'l1': 5, 'giou' : 2}

    actor = actors.MyActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad], "lr": 2e-5}
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    trainer.train(120, load_latest=True, fail_safe=True)
