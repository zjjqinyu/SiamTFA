import os
import os.path
from glob import glob
import numpy as np
import pyarrow as pa
import torch
# import pandas
from copy import deepcopy
import random
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader, opencv_loader
from ltr.admin.environment import env_settings


class BaseRgbtDataset(BaseVideoDataset):
    """ BaseRgbtDataset.
    """

    def __init__(self, name, root, image_loader, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name):
        """
        args:
            name - dataset name
            root - path to the dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
        """
        super().__init__(name, root, image_loader)
        self.rgb_sub_dir = rgb_sub_dir
        self.aux_sub_dir = aux_sub_dir
        self.rgb_anno_name = rgb_anno_name
        self.aux_anno_name = aux_anno_name

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()
        self.frame_path_map = self._build_frame_path_map()

    def get_name(self):
        return self.name
    
    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _build_frame_path_map(self):
        seq_frame_map = []
        for seq_name in self.sequence_list:
            rgb_img_name_list = sorted([os.path.basename(p) for p in glob(os.path.join(self.root, seq_name, self.rgb_sub_dir, '*')) if os.path.splitext(p)[1] in ['.png', '.bmp', '.jpg', '.jpeg']])
            aux_img_name_list = sorted([os.path.basename(p) for p in glob(os.path.join(self.root, seq_name, self.aux_sub_dir, '*')) if os.path.splitext(p)[1] in ['.png', '.bmp', '.jpg', '.jpeg']])
            img_name_list_len = min(len(rgb_img_name_list), len(aux_img_name_list))
            rgb_img_name_list = rgb_img_name_list[:img_name_list_len]
            aux_img_name_list = aux_img_name_list[:img_name_list_len]

            rgb_anno_path = os.path.join(self.root, seq_name, self.rgb_anno_name)
            aux_anno_path = os.path.join(self.root, seq_name, self.aux_anno_name)

            rgb_anno_list= self._load_anno(rgb_anno_path)
            aux_anno_list= self._load_anno(aux_anno_path)

            rgb_anno_list = np.array(rgb_anno_list, dtype=np.string_)
            aux_anno_list = np.array(aux_anno_list, dtype=np.string_)

            assert len(rgb_anno_list)>=img_name_list_len, rgb_anno_path
            assert len(aux_anno_list)>=img_name_list_len, aux_anno_path
            rgb_anno_list= rgb_anno_list[:img_name_list_len]
            aux_anno_list= aux_anno_list[:img_name_list_len]

            # rgb_img_name:0  aux_img_name:1
            seq_frame_map.append(list(zip(rgb_img_name_list, aux_img_name_list)))
        return seq_frame_map
    
    def _get_sequence_list(self):
        return sorted([p for p in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, p))])
    
    def _load_anno(self, anno_path):
        try:
            anno_array= np.loadtxt(anno_path, dtype=np.int32)
        except:
            try:
                anno_array = np.loadtxt(anno_path, delimiter=',', dtype=np.int32)
            except Exception as e:
                print(e)
                # print(anno_path)
        anno_array = self._anno_pre_proc(anno_array)
        assert not (anno_array[:, 2:]<=0).all(), anno_path
        return anno_array
    
    def _anno_pre_proc(self, anno_array):
        return anno_array

    def _read_rgb_bb_anno(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        anno = self._load_anno(os.path.join(seq_path, self.rgb_anno_name))
        return torch.tensor(anno)

    def _read_aux_bb_anno(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        anno =  self._load_anno(os.path.join(seq_path, self.aux_anno_name))
        return torch.tensor(anno)

    def _read_target_visible(self, seq_id):
        # For dataset without visible labels, the default setting is no occlusion
        frames_len = len(self.frame_path_map[seq_id])
        occlusion = torch.ByteTensor([0]*frames_len)
        cover = torch.ByteTensor([8]*frames_len)
        target_visible = ~occlusion & (cover>0).byte()
        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_rgb_frame_path(self, seq_id, frame_id):
        seq_path = self._get_sequence_path(seq_id)
        frame_name = self.frame_path_map[seq_id][frame_id][0]
        return os.path.join(seq_path, self.rgb_sub_dir, frame_name)
    
    def _get_aux_frame_path(self, seq_id, frame_id):
        seq_path = self._get_sequence_path(seq_id)
        frame_name = self.frame_path_map[seq_id][frame_id][1]
        return os.path.join(seq_path, self.aux_sub_dir, frame_name)

    def _get_rgb_frame(self, seq_id, frame_id):
        return self.image_loader(self._get_rgb_frame_path(seq_id, frame_id))

    def _get_aux_frame(self, seq_id, frame_id):
        return self.image_loader(self._get_aux_frame_path(seq_id, frame_id))
    
    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        rgb_bbox = self._read_rgb_bb_anno(seq_id)
        aux_bbox = self._read_aux_bb_anno(seq_id)
        valid = (rgb_bbox[:, 2] > 0) & (rgb_bbox[:, 3] > 0) & (aux_bbox[:, 2] > 0) & (aux_bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_id)
        assert min(len(rgb_bbox), len(aux_bbox)) >= len(visible), seq_path
        valid = valid[:len(visible)]
        visible_ratio = visible_ratio[:len(visible)]
        visible = visible & valid.byte()
        return {'rgb_bbox': rgb_bbox, 'aux_bbox': aux_bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        rgb_frame_list = [self._get_rgb_frame(seq_id, f_id) for f_id in frame_ids]
        aux_frame_list = [self._get_aux_frame(seq_id, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return rgb_frame_list, aux_frame_list, anno_frames

class RGBT234(BaseRgbtDataset):
    """ RGBT234 dataset.
        The file organization of the dataset is like:
        '<dataset_root_dir>/<seq_1>',
        '<dataset_root_dir>/<seq_2>',
        ...
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None):
        name = 'RGBT234'
        if root is None:
            root = os.path.join(env_settings().rgbt234_dir)
        rgb_sub_dir = 'visible'
        aux_sub_dir = 'infrared'
        rgb_anno_name = 'visible.txt'
        aux_anno_name = 'infrared.txt'
        super().__init__(name, root, image_loader, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)

class LasHeR(BaseRgbtDataset):
    """ LasHeR dataset.
        The file organization of the dataset is like:
        '<dataset_root_dir>/train/<seq_1>',
        ...
        '<dataset_root_dir>/test/<seq_1>',
        ...
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train'):
        assert split in ['train', 'test']
        name = 'LasHeR_'+split
        if root is None:
            root = os.path.join(env_settings().lasher_dir, split)
        rgb_sub_dir = 'visible'
        aux_sub_dir = 'infrared'
        rgb_anno_name = 'visible.txt'
        aux_anno_name = 'infrared.txt'
        super().__init__(name, root, image_loader, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)

class GTOT(BaseRgbtDataset):
    """ GTOT dataset.
        The file organization of the dataset is like:
        '<dataset_root_dir>/<seq_1>',
        '<dataset_root_dir>/<seq_2>',
        ...
    """
    def __init__(self, root=None, image_loader=opencv_loader, split=None):
        name = 'GTOT'
        if root is None:
            root = os.path.join(env_settings().gtot_dir)
        rgb_sub_dir = 'v'
        aux_sub_dir = 'i'
        rgb_anno_name = 'groundTruth_v.txt'
        aux_anno_name = 'groundTruth_i.txt'
        super().__init__(name, root, image_loader, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)

    def _anno_pre_proc(self, anno_array):
        anno_array[:, 2] = anno_array[:, 2] - anno_array[:, 0]
        anno_array[:, 3] = anno_array[:, 3] - anno_array[:, 1]
        return anno_array


class VTUAV(BaseRgbtDataset):
    """ VTUAV dataset.
        The file organization of the dataset is like:
        '<dataset_root_dir>/train_ST/<seq_1>',
        ...
        '<dataset_root_dir>/train_LT/<seq_1>',
        ...
        '<dataset_root_dir>/test_ST/<seq_1>',
        ...
        '<dataset_root_dir>/test_LT/<seq_1>',
        ...
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train_ST'):
        assert split in ['train_ST', 'train_LT', 'test_ST', 'test_LT']
        name = 'VTUAV_'+split
        if root is None:
            root = os.path.join(env_settings().vtuav_dir, split)
        rgb_sub_dir = 'rgb'
        aux_sub_dir = 'ir'
        rgb_anno_name = 'rgb.txt'
        aux_anno_name = 'ir.txt'
        super().__init__(name, root, image_loader, rgb_sub_dir, aux_sub_dir, rgb_anno_name, aux_anno_name)

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


