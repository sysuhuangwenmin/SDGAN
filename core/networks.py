from torch import nn
import torch
import torch.nn.functional as F
from style.utils import FixedStyledConv,EqualConv2d,EqualLinear,StyledConv,ConvLayer,Upsample

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
from utils import weights_init
import math
import matplotlib.pyplot as plt
import torchvision.utils as vutils
def Visi(img_a,name):
    vutils.save_image(
        img_a, name,
        nrow=1, normalize=True, range=(-1., 1.)
    )
def visualize_alpha(output_name, tensor):
    tensor = tensor.detach().cpu().permute(1,2,0).numpy()
    fig = plt.matshow(tensor, cmap="inferno")
    plt.gca().set_axis_off()
    plt.gcf().set_dpi(100)
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)

##################################################################################
# Generator
##################################################################################



class Gen(nn.Module):  # M、F、E、T、D
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters['tags']
        self.skip = hyperparameters['skip']

        self.style_dim = hyperparameters['style_dim']
        self.noise_dim = hyperparameters['noise_dim']  # 32

        channels = hyperparameters['encoder']['channels']  # [64, 128, 256]
        self.encoder = nn.Sequential(
            EqualConv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )


        channels = hyperparameters['decoder']['channels']  # [256, 128, 64]
        self.decoder = nn.Sequential(
            *[UpBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            EqualConv2d(channels[-1], hyperparameters['input_dim'], 1, 1, 0)
        )

        self.extractors = Extractors(hyperparameters)

        self.translators = nn.ModuleList([Translator(hyperparameters)
                                          for i in range(len(self.tags))]
                                         )
        self.translator=Translator_(hyperparameters)

        self.mappers = nn.ModuleList([Mapper(hyperparameters, len(self.tags[i]['attributes']))
                                      for i in range(len(self.tags))]
                                     )

        self.out = EqualConv2d(64, 256, 1, 1, 0)

    def encode(self, x):
        e=self.encoder(x)
        return e

    def decode(self, e):  #256*64*64
        x = self.decoder(e)
        return x

    def extract(self, x, i):
        return self.extractors(x)

    def map(self, z, i, j):
        return self.mappers[i](z, j)

    def translate(self, e, s_i, s_r, tag):

        if tag == 0:
            f1,m_b,m_b_delta = self.translators[0](e, s_r)
            f2,m_e,m_e_delta = self.translators[1](e, s_i[:, 1])
            f3,m_h,m_h_delta = self.translators[2](e, s_i[:, 2])
            f4,m_g,m_g_delta = self.translator(e)

        elif tag == 1:
            f1, m_b, m_b_delta = self.translators[0](e, s_i[:,0])
            f2, m_e, m_e_delta = self.translators[1](e, s_r)
            f3, m_h, m_h_delta = self.translators[2](e, s_i[:, 2])
            f4, m_g, m_g_delta = self.translator(e)

        else:
            f1, m_b, m_b_delta = self.translators[0](e, s_i[:, 0])
            f2, m_e, m_e_delta = self.translators[1](e, s_i[:, 1])
            f3, m_h, m_h_delta = self.translators[2](e, s_r)
            f4, m_g, m_g_delta = self.translator(e)


        mask = F.softmax(torch.cat([m_b, m_e, m_h, m_g], dim=1), dim=1)
        mask_delta = torch.cat([m_b_delta, m_e_delta, m_h_delta, m_g_delta], dim=1)

        f_delta = f1 * mask_delta[:, 0].unsqueeze(1) + f2 * mask_delta[:, 1].unsqueeze(1) + \
                  f3 * mask_delta[:, 2].unsqueeze(1) + f4 * mask_delta[:, 3].unsqueeze(1)
        f = self.out(f_delta)

        if self.skip:
            return e * (1 - mask_delta[:, tag].unsqueeze(1)) + f * mask_delta[:, tag].unsqueeze(1)

        return f

##################################################################################
# Extractors, Translator and Mapper
##################################################################################
class Extractors(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.num_tags = len(hyperparameters['tags'])
        channels = hyperparameters['extractors']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1],  hyperparameters['style_dim'] * (self.num_tags), 1, 1, 0),
        )

    def forward(self, x):
        s = self.model(x).view(x.size(0), self.num_tags, -1)
        return s

class Translator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']  # [64, 64, 64, 64, 64, 64, 64, 64]
        self.model = nn.Sequential(EqualConv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[MiddleBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
                                   EqualConv2d(64,64,1,1,0)
        )

        self.mask = nn.ModuleList([EqualConv2d(64, 1, 1, 1, 0),EqualConv2d(256, 1, 1, 1, 0)])
        self.mask_delta = nn.ModuleList([nn.Sequential(nn.Tanh(), EqualConv2d(64, 1, 1, 1, 0)),
                                   nn.Sequential(nn.Tanh(), EqualConv2d(256, 1, 1, 1, 0))])
        self.style_to_params = EqualLinear(hyperparameters['style_dim'],
                                         self.get_num_adain_params(self.model))  # 256*（7*256）

        self.features = nn.Sequential(
            EqualConv2d(channels[-1], hyperparameters['decoder']['channels'][0], 3, 1, 1),
        )


    def forward(self, e, s):
        p = self.style_to_params(s)
        self.assign_adain_params(p, self.model)

        mid = self.model(e)  # 8*64*64*64
        mask_a = self.mask[0](mid.detach())
        mask_b = self.mask[1](e.detach())
        mask=mask_a+mask_b
        delta_mask_a = self.mask_delta[0](mid)
        delta_mask_b = self.mask_delta[1](e)
        delta_mask=delta_mask_a+delta_mask_b
        delta_mask = F.sigmoid(delta_mask)

        return mid, mask, delta_mask

    def assign_adain_params(self, adain_params, model):  # 8*1792  给AdaIN块分配尺度和偏置
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.weight = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features,1)
                m.bias = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features,1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2*m.num_features
        return num_adain_params  # 7*256

class Translator_(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']

        self.model = nn.Sequential(EqualConv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[KeepBlockIN(channels[i]) for i in range(len(channels) - 1)],
                                   EqualConv2d(64, 64, 1, 1, 0)
        )
        self.mask = nn.ModuleList([EqualConv2d(64, 1, 1, 1, 0),EqualConv2d(256, 1, 1, 1, 0)])
        self.mask_delta = nn.ModuleList([nn.Sequential(nn.Tanh(), EqualConv2d(64, 1, 1, 1, 0)),
                                   nn.Sequential(nn.Tanh(), EqualConv2d(256, 1, 1, 1, 0))])


        self.features = nn.Sequential(
            EqualConv2d(channels[-1], hyperparameters['decoder']['channels'][0], 3, 1, 1),
        )


    def forward(self, e):


        mid = self.model(e)
        mask_a = self.mask[0](mid.detach())
        mask_b = self.mask[1](e.detach())
        mask=mask_a+mask_b
        delta_mask_a = self.mask_delta[0](mid)
        delta_mask_b = self.mask_delta[1](e)
        delta_mask=delta_mask_a+delta_mask_b
        delta_mask = F.sigmoid(delta_mask)

        return mid, mask, delta_mask


class Mapper(nn.Module):
    def __init__(self, hyperparameters, num_attributes):
        super().__init__()
        channels = hyperparameters['mappers']['pre_channels']
        self.pre_model = nn.Sequential(
            nn.Linear(hyperparameters['noise_dim'], channels[0]),
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hyperparameters['mappers']['post_channels']
        self.post_models = nn.ModuleList([nn.Sequential(
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Linear(channels[-1], hyperparameters['style_dim']),
            ) for i in range(num_attributes)
        ])

    def forward(self, z, j):
        z = self.pre_model(z)
        return self.post_models[j](z)


##################################################################################
# Basic Blocks
##################################################################################

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = FixedStyledConv(in_dim, in_dim, 3)
        self.conv2 = FixedStyledConv(in_dim, out_dim, 3, downsample=True)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.

        self.sc = ConvLayer(
            in_dim, out_dim, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, x):
        residual =self.sc(x)
        out = self.conv2(self.conv1(x))
        out = residual + out
        return out / math.sqrt(2)


class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = FixedStyledConv(in_dim, out_dim, 3, upsample=True)
        self.conv2 = FixedStyledConv(out_dim, out_dim, 3)


        self.sc = EqualConv2d(in_dim, out_dim, 1, 1, 0)
        self.upsample = Upsample([1, 3, 3, 1])

    def forward(self, x):
        residual = self.upsample(self.sc(x))
        out = self.conv2(self.conv1(x))
        out = residual + out
        return out / math.sqrt(2)



class KeepBlockIN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv1 = FixedStyledConv(in_dim, in_dim, 3)
        #self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.sc = EqualConv2d(in_dim, in_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv1(x)
        out = residual + out
        return out / math.sqrt(2)



class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(self.adain1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))

##################################################################################
# Basic Modules and Functions
##################################################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.bias = None
        self.weight = None

    def forward(self, x):
        assert self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

