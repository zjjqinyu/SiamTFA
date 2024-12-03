import torch
from torch import nn
from torchvision.models import swin_b, Swin_B_Weights

class JcfaMoudle(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(JcfaMoudle, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                    nn.BatchNorm2d(out_channels),
                    nn.AvgPool2d(kernel_size=stride, stride=stride))
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (n,c1,1,1)
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=1, stride=1, groups=out_channels*2),  # (n,c1,1,1)
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*3, kernel_size=1, stride=1),   # (n,c2,1,1)
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1, groups=out_channels*2), # (n,c,h,w)
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels*2, out_channels=3, kernel_size=1, stride=1),  # (n,1,h,w)
            nn.Sigmoid()
        )
        self.fusion_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (n,c1,1,1)
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=1, stride=1, groups=out_channels*2),  # (n,c1,1,1)
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=1, stride=1),   # (n,c2,1,1)
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, aux_feat, fus_feat):
        '''forward
        Args:
            rgb_feat (tensor): shape: (n, c, h, w) 
            aux_feat (tensor): shape: (n, c, h, w) 
            fus_feat (tensor): shape: (n, c, h, w) 
        '''
        modal_concat_feat = torch.concat((rgb_feat, aux_feat), dim=1) # (n, 2c, h, w)

        w_channel_attn_com, w_channel_attn_rgb, w_channel_attn_aux = torch.chunk(self.channel_attn(modal_concat_feat), 3, 1)
        rgb_feat_attn = rgb_feat * (1+w_channel_attn_com) * (1+w_channel_attn_rgb)
        aux_feat_attn = aux_feat * (1+w_channel_attn_com) * (1+w_channel_attn_aux)

        w_spatial_attn_com, w_spatial_attn_rgb, w_spatial_attn_aux = torch.chunk(self.spatial_attn(modal_concat_feat), 3, 1)
        rgb_feat_attn = rgb_feat_attn * (1+w_spatial_attn_com) * (1+w_spatial_attn_rgb)
        aux_feat_attn = aux_feat_attn * (1+w_spatial_attn_com) * (1+w_spatial_attn_aux)
        new_fus_feat = rgb_feat_attn + aux_feat_attn
        new_fus_feat = nn.functional.layer_norm(new_fus_feat, new_fus_feat.shape[1:])

        last_fus_feat = self.conv1(fus_feat)
        fusion_concat_feat = torch.concat((last_fus_feat, new_fus_feat), dim=1) # (n, 2c, h, w)
        w_fusion_channel_attn = self.fusion_channel_attn(fusion_concat_feat)
        fusion_concat_feat = fusion_concat_feat*(1+w_fusion_channel_attn)
        fus_feat1, fus_feat2 = torch.chunk(fusion_concat_feat, 2, 1)
        out = fus_feat1 + fus_feat2
        out = nn.functional.layer_norm(out, out.shape[1:])
        return out

class TfaSwinBackbone(nn.Module):
    def __init__(self, weights=None):
        super(TfaSwinBackbone, self).__init__()
        feature_extractor_rgb = swin_b(weights=weights).features
        self.rgb_stage_list = nn.ModuleList([feature_extractor_rgb[i: i+2] for i in range(0, 8, 2)])

        feature_extractor_aux = swin_b(weights=weights).features
        self.aux_stage_list = nn.ModuleList([feature_extractor_aux[i: i+2] for i in range(0, 8, 2)])

        in_channels_list = [6, 128, 256, 512]
        out_channels_list = [128, 256, 512, 1024]
        stride_list = [4, 2, 2, 2]
        self.fus_stage_list = nn.ModuleList([JcfaMoudle(in_channels, out_channels, stride)
                                             for in_channels, out_channels, stride in zip(in_channels_list, out_channels_list, stride_list)])

    def forward(self, rgb_img, aux_img):
        rgb_feat = rgb_img
        aux_feat = aux_img
        fus_feat = torch.concat((rgb_img, aux_img), dim=1) # (n,c,h,w)
        for i in range(len(self.fus_stage_list)):
            rgb_feat = self.rgb_stage_list[i](rgb_feat) # (n, h, w, c) 
            aux_feat = self.aux_stage_list[i](aux_feat) # (n, h, w, c) 
            fus_feat = self.fus_stage_list[i](rgb_feat.permute(0, 3, 1, 2), aux_feat.permute(0, 3, 1, 2), fus_feat)  # input and output: (n, c, h, w) 
        return fus_feat

def tfa_swin_backbone(weights=Swin_B_Weights.IMAGENET1K_V1):
    return TfaSwinBackbone(weights)