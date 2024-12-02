import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None, joint_transform=None, rgb_transform=None, aux_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
            rgb_transform - The set of transformations to be applied on the rgb images. 
            aux_transform - The set of transformations to be applied on the aux images. 
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'template':  transform if template_transform is None else template_transform,
                          'joint': joint_transform,
                          'rgb': rgb_transform,
                          'aux': aux_transform
                          }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class MultimodalProcessing(BaseProcessing):
    """ The processing class used for Multimodal training. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """

    def __init__(self, template_area_factor, template_sz, search_area_factor, search_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', *args, **kwargs):
        """
        args:
            template_area_factor - The size of the template region  relative to the target size.
            template_sz - An integer, denoting the size to which the template region is resized. The template region is always
                        square.
            search_area_factor - The size of the search region  relative to the target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.template_area_factor = template_area_factor
        self.template_sz = template_sz
        self.search_area_factor = search_area_factor
        self.search_sz = search_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode, modal):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data
            modal - string 'rgb' or 'aux' indicating rgb or aux data

        returns:
            torch.Tensor - jittered box
        """
        if modal == 'rgb':
            self.jittered_r1 = torch.randn(2)
            self.jittered_r2 = torch.rand(2)
        jittered_size = box[2:4] * torch.exp(self.jittered_r1 * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (self.jittered_r2 - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_rgb_images', 'template_aux_images', 'template_rgb_anno', 'template_aux_anno', 
                'search_rgb_images', 'search_aux_images', 'search_rgb_anno', 'search_aux_anno' 
        returns:
            TensorDict - output data block with following fields:
                'template_rgb_images', 'template_aux_images', 'template_rgb_anno', 'template_aux_anno', 
                'search_rgb_images', 'search_aux_images', 'search_rgb_anno', 'search_aux_anno' 
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_rgb_images'], data['template_rgb_anno'] = self.transform['joint'](image=data['template_rgb_images'], bbox=data['template_rgb_anno'])
            data['template_aux_images'], data['template_aux_anno'] = self.transform['joint'](image=data['template_aux_images'], bbox=data['template_aux_anno'], new_roll=False)
            data['search_rgb_images'], data['search_rgb_anno'] = self.transform['joint'](image=data['search_rgb_images'], bbox=data['search_rgb_anno'], new_roll=False)
            data['search_aux_images'], data['search_aux_anno'] = self.transform['joint'](image=data['search_aux_images'], bbox=data['search_aux_anno'], new_roll=False)
        
        # Apply rgb and aux transforms
        for modal in ['rgb', 'aux']:
            for s in ['template', 'search']:
                if self.transform[modal] is not None:
                    data[s + '_' + modal + '_images'], data[s + '_' + modal + '_anno'] = self.transform[modal](image=data[s + '_' + modal + '_images'], bbox=data[s + '_' + modal + '_anno'], joint=False, new_roll=(s=='template'))

        # Apply template and search transforms
        for s in ['template', 'search']:
            for modal in ['rgb', 'aux']:
                assert self.mode == 'sequence' or len(data[s + '_' + modal + '_images'][modal]) == 1, \
                    "In pair mode, num template/search frames must be 1"

                # Add a uniform noise to the center pos
                jittered_anno = [self._get_jittered_box(a, s, modal) for a in data[s + '_' + modal + '_anno']]

                # Crop image region centered at jittered_anno box
                if s == 'template':
                    crops, boxes, _ = prutils.jittered_center_crop(data[s + '_' + modal + '_images'], jittered_anno, data[s + '_' + modal + '_anno'],
                                                                self.template_area_factor, self.template_sz)
                elif s =='search':
                    crops, boxes, _ = prutils.jittered_center_crop(data[s + '_' + modal + '_images'], jittered_anno, data[s + '_' + modal + '_anno'],
                                                                self.search_area_factor, self.search_sz)
                else:
                    raise KeyError

                # Apply transforms
                data[s + '_' + modal + '_images'], data[s + '_' + modal + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False, new_roll=(modal=='rgb'))

                # anno: (x,y,h,w) -> norm(cx,cy,h,w)
                for i, _ in enumerate(data[s + '_' + modal + '_anno']):
                    data[s + '_' + modal + '_anno'][i] = data[s + '_' + modal + '_anno'][i]/crops[i].shape[0]   # norm(x, y, h, w)
        
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].squeeze()
        return data # norm(x, y, h, w)