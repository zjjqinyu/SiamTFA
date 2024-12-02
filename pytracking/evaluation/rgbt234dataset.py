from pytracking.evaluation.data import RgbtBaseDataset
from pytracking.evaluation.environment import env_settings

class RGBT234Dataset(RgbtBaseDataset):
    def __init__(self, dtype):
        base_path = env_settings().rgbt234_path
        dataset_name = 'RGBT234'
        rgb_sub_dir = 'visible'
        aux_sub_dir = 'infrared'
        rgb_anno_name = 'visible.txt'
        aux_anno_name = 'infrared.txt'
        super().__init__(dtype, base_path, dataset_name, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)
