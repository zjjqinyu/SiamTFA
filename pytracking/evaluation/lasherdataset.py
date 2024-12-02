import os
from pytracking.evaluation.data import RgbtBaseDataset
from pytracking.evaluation.environment import env_settings

class LasHeRDataset(RgbtBaseDataset):
    def __init__(self, dtype, split='test'):
        assert split in ['test', 'train']
        base_path = os.path.join(env_settings().lasher_path, split)
        dataset_name = 'LasHeR_'+split
        rgb_sub_dir = 'visible'
        aux_sub_dir = 'infrared'
        rgb_anno_name = 'visible.txt'
        aux_anno_name = 'infrared.txt'
        super().__init__(dtype, base_path, dataset_name, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)
