import torch
from torch import nn

class PixelwiseXcorr(nn.Module):
    def __init__(self, template_feat_sz, search_feat_sz, out_channels):
        super(PixelwiseXcorr, self).__init__()
        self.end_1x1_conv = nn.Conv2d(in_channels=template_feat_sz**2, out_channels=out_channels, kernel_size=1, stride=1)
        self.norm_layer = nn.LayerNorm(search_feat_sz)

    def forward(self, template_feat, search_feat):
        funsion_feat = self._pixelwise_xcorr(kernel=template_feat, search=search_feat)   # c=template_feat_sz**2; h,w=search_feat_sz
        funsion_feat = self.norm_layer(funsion_feat)
        funsion_feat = self.end_1x1_conv(funsion_feat) # (n,c,h,w) c=out_channels; h,w=search_feat_sz
        return funsion_feat

    def _pixelwise_xcorr(self, kernel, search):
        n, c, h, w = search.shape
        ker = kernel.reshape(n, c, -1).transpose(1, 2)
        feat = search.reshape(n, c, -1)
        corr = torch.matmul(ker, feat)
        corr = corr.reshape(*corr.shape[:2], h, w)
        return corr
    
def pixelwise_xcorr(template_feat_sz, search_feat_sz, out_channels):
    return PixelwiseXcorr(template_feat_sz, search_feat_sz, out_channels)