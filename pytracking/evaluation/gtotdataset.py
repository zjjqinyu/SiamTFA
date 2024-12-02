from pytracking.evaluation.data import RgbtBaseDataset
from pytracking.evaluation.environment import env_settings

class GTOTDataset(RgbtBaseDataset):
    def __init__(self, dtype):
        base_path = env_settings().gtot_path
        dataset_name = 'GTOT'
        rgb_sub_dir = 'v'
        aux_sub_dir = 'i'
        rgb_anno_name = 'groundTruth_v.txt'
        aux_anno_name = 'groundTruth_i.txt'
        super().__init__(dtype, base_path, dataset_name, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)

    def _anno_pre_proc(self, anno_array):
        anno_array[:, 2] = anno_array[:, 2] - anno_array[:, 0]
        anno_array[:, 3] = anno_array[:, 3] - anno_array[:, 1]
        return anno_array
