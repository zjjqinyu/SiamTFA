import os
from pytracking.evaluation.data import RgbtBaseDataset
from pytracking.evaluation.environment import env_settings

class VTUAVDataset(RgbtBaseDataset):
    def __init__(self, dtype, split='test_ST'):
        assert split in ['test_ST', 'test_LT', 'train_ST', 'train_LT']
        base_path = os.path.join(env_settings().vtuav_path, split)
        dataset_name = 'VTUAV_' + split
        rgb_sub_dir = 'rgb'
        aux_sub_dir = 'ir'
        rgb_anno_name = 'rgb.txt'
        aux_anno_name = 'ir.txt'
        super().__init__(dtype, base_path, dataset_name, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)
    
    def _anno_pre_proc(self, anno_array):
        anno_array_r = anno_array.repeat(10, axis=0)
        for i in range(len(anno_array)-1):
               temp = anno_array[i:i+2, 2:]
               if temp.all() > 0:
                    self._fill_anno_smooth(anno_array_r, i*10, (i+1)*10)
        return anno_array_r

    def _fill_anno_smooth(self, arr, start_idx, end_idx):
        global _arr
        _arr = arr
        if end_idx - start_idx > 1:
            mid_idx = (start_idx+end_idx)//2
            _arr[mid_idx, :] = (_arr[start_idx, :] + _arr[end_idx, :])//2
            self._fill_anno_smooth(_arr, start_idx, mid_idx)
            self._fill_anno_smooth(_arr, mid_idx, end_idx)
