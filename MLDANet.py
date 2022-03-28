import torch
import torch.nn as nn
from torchvision.models import vgg16
from models.position_encoding import build_position_encoding
from models.deformable_transformerffn import DeformableTransformer_CD,DeformableTransformer_CD_otherskips
import torch.nn.functional as F
from torch.autograd import Variable
class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        vggmodel=vgg16(pretrained=True).cuda()
        features = list(vggmodel.features)[:30]
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {8,15,22,29}:
                results.append(x)
        return results
def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm2d(inplanes,affine=True)

    return out
def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out
class Conv2d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(Conv2d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv2x2x2(in_planes, out_planes, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv2d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1),padding=(0, 0),dilation=(1, 1),bias=False,weight_std=False):
        super(Conv2dBlock,self).__init__()
        self.conv = conv2x2x2(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv2dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv2dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out
class deformable_transformer_CD_net(nn.Module):
    def __init__(self,norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=1, weight_std=False,num_encoder_layers=1,num_enc_n_points=4,num_n_heads=8):
        super(deformable_transformer_CD_net,self).__init__()
        self.MODEL_NUM_CLASSES=num_classes
        self.backbone=vgg16_base()
        self.position_embed64=build_position_encoding(mode='v2',hidden_dim=64)
        self.encoder_Detrans_ms=DeformableTransformer_CD(d_model=64,dim_feedforward=1024,num_encoder_layers=num_encoder_layers,enc_n_points=num_enc_n_points,nhead=num_n_heads,num_feature_levels=4)
        self.conv_stage0=nn.Conv2d(128,64,kernel_size=3,padding=1)

        # self.upsamplex2=nn.Upsample(scale_factor=(2,2),mode="bilinear")
        self.cls_conv = nn.Conv2d(32, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.sigmoid=nn.Sigmoid()
        self.conv512to64s1=nn.Conv2d(1024,64,1)
        self.conv512to64s1=nn.Conv2d(1024,64,1)
        self.conv256to64s1=nn.Conv2d(512,64,1)
        self.conv128to64s1=nn.Conv2d(256,64,1)
        self.trconv1=nn.ConvTranspose2d(64,64,2,2)
        self.trconv2=nn.ConvTranspose2d(64,64,2,2)
        self.trconv3=nn.ConvTranspose2d(64,64,2,2)
        self.trconv4=nn.ConvTranspose2d(64,32,2,2)
        self.coefficient1=nn.Parameter(torch.Tensor([1.]))
        self.coefficient2=nn.Parameter(torch.Tensor([1.]))
        self.coefficient3=nn.Parameter(torch.Tensor([1.]))
        self.coefficient4=nn.Parameter(torch.Tensor([1.]))
    def posi_mask(self,x):
        x_fea=[]
        x_posemb=[]
        masks=[]
        for lvl,fea in enumerate(x):
            x_fea.append(fea)
            x_posemb.append(self.position_embed64(fea))
            masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3]), dtype=torch.bool).cuda())
        return x_fea,masks,x_posemb
    def forward(self,input1,input2):
        x_conv=[]
        x_convs1=self.backbone(input1)
        x_convs2=self.backbone(input2)
        diff=[]
        trans_out=[]
        for fea1,fea2 in zip(x_convs1,x_convs2):
            fea=torch.cat((fea1,fea2),1)
            x_conv.append(fea)
        diff.append(self.conv128to64s1(x_conv[0]))
        diff.append(self.conv256to64s1(x_conv[1]))
        diff.append(self.conv512to64s1(x_conv[2]))
        diff.append(self.conv512to64s1(x_conv[-1]))
        
        x_fea,masks,x_posemb=self.posi_mask(diff)
        
        x_trans=self.encoder_Detrans_ms(x_fea,masks,x_posemb)
        
        trans_out.append(x_trans[:,0:16384].transpose(-1,-2).view(diff[-4].shape))
        trans_out.append(x_trans[:,16384:20480].transpose(-1,-2).view(diff[-3].shape))
        trans_out.append(x_trans[:,20480:21504].transpose(-1,-2).view(diff[-2].shape))
        trans_out.append(x_trans[:,21504::].transpose(-1,-2).view(diff[-1].shape))


    
        out=self.trconv1(trans_out[-1])
        out=out+trans_out[-2]
        out=self.trconv2(out)
        out=out+trans_out[-3]
        out=self.trconv3(out)
        out=out+trans_out[-4]
        out=self.trconv4(out)
        result=self.cls_conv(out)
        result=self.sigmoid(result)

        return result

