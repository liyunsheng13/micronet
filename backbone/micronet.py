
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.activation as activation
import backbone.microconfig as microcfg


import math
import pdb

__all__ = ['MicroNet', 'micronet']

TAU = 20
#####################################################################3
# part 1: functions
#####################################################################3

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

def conv_3x3_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def gcd(a, b):
    a, b = (a, b) if a >= b else (b, a)
    while b:
        a, b = b, a%b
    return a

#####################################################################3
# part 2: modules
#####################################################################3

class SwishLinear(nn.Module):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)

class StemLayer(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, mode='default', groups=(4,4)):
        super(StemLayer, self).__init__()

        self.exp = 1 if mode == 'default' else 2
        g1, g2 = groups 
        if mode == 'default':
            self.stem = nn.Sequential(
                nn.Conv2d(inp, oup*self.exp, 3, stride, 1, bias=False, dilation=dilation),
                nn.BatchNorm2d(oup*self.exp),
                nn.ReLU6(inplace=True) if self.exp == 1 else MaxGroupPooling(self.exp)
            )
        elif mode == 'spatialsepsf':
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.ReLU6(inplace=True)
            )
        else: 
            exp = 6
            ch_per_group=2
            hidden_dim = inp*exp //ch_per_group
            self.stem = nn.Sequential(
                DepthExpandConv(inp, exp, kernel_size=3, stride=stride),
                MaxGroupPooling(ch_per_group),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, groups=1),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
           
    def forward(self, x):
        out = self.stem(x)    
        return out

class GroupConv(nn.Module):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        print ('inp: %d, oup:%d, g:%d' %(inp, oup, self.groups[0]))
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=self.groups[0]),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out

class ChannelShuffle2(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle2, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)

        return out

######################################################################3
# part 3: new block
#####################################################################3

class SpatialSepConvSF(nn.Module):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias=False, groups=1
            ),
            nn.BatchNorm2d(oup1),
            nn.Conv2d(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=oup1
            ),
            nn.BatchNorm2d(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Module):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp*exp1
        oup = inp*exp1*exp2
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp*exp1, 
                (kernel_size, 1), 
                (stride, 1), 
                (kernel_size//2, 0), 
                bias=False, groups=inp
            ),
            nn.BatchNorm2d(inp*exp1),
            nn.Conv2d(hidden_dim, oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=hidden_dim
            ),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def get_pointwise_conv(mode, inp, oup, hiddendim, groups):

    if mode == 'group':
        return GroupConv(inp, oup, groups)
    elif mode == '1x1':
        return nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup)
                )
    else:
        return None
 
class DYMicroBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, ch_exp=(2, 2), ch_per_group=4, groups_1x1=(1, 1), depthsep=True, shuffle=False, pointwise='fft', activation_cfg=None):
        super(DYMicroBlock, self).__init__()

        print(activation_cfg.dy)

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg.dy
        act = activation_cfg.MODULE
        act_max = activation_cfg.ACT_MAX
        act_bias = activation_cfg.LINEARSE_BIAS
        act_reduction = activation_cfg.REDUCTION * activation_cfg.ratio
        init_a = activation_cfg.INIT_A
        init_b = activation_cfg.INIT_B
        init_ab3 = activation_cfg.INIT_A_BLOCK3

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                activation.get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle2(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                activation.get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                activation.get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                activation.get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                activation.get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle2(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle2(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)), #FFTConv
                activation.get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle2(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out

###########################################################################

class MicroNet(nn.Module):
    def __init__(self, cfg, input_size=224, num_classes=1000, teacher=False):
        super(MicroNet, self).__init__()
        # setting of inverted residual blocks

        mode = cfg.MODEL.MICRONETS.NET_CONFIG
        self.cfgs = microcfg.get_micronet_config(mode)

        block = eval(cfg.MODEL.MICRONETS.BLOCK)
        stem_mode = cfg.MODEL.MICRONETS.STEM_MODE
        stem_ch = cfg.MODEL.MICRONETS.STEM_CH
        stem_dilation = cfg.MODEL.MICRONETS.STEM_DILATION
        stem_groups = cfg.MODEL.MICRONETS.STEM_GROUPS
        out_ch = cfg.MODEL.MICRONETS.OUT_CH
        depthsep = cfg.MODEL.MICRONETS.DEPTHSEP
        shuffle = cfg.MODEL.MICRONETS.SHUFFLE
        pointwise = cfg.MODEL.MICRONETS.POINTWISE
        dropout_rate = cfg.MODEL.MICRONETS.DROPOUT

        act_max = cfg.MODEL.ACTIVATION.ACT_MAX
        act_bias = cfg.MODEL.ACTIVATION.LINEARSE_BIAS
        activation_cfg= cfg.MODEL.ACTIVATION

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation, 
                    mode=stem_mode,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)


        self.avgpool = nn.Sequential(
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        ) 

        # building last several layers
        output_channel = out_ch
         
        self.classifier = nn.Sequential(
            SwishLinear(input_channel, output_channel),
            nn.Dropout(dropout_rate),
            SwishLinear(output_channel, num_classes)
        )
        self._initialize_weights()
           
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def micronet(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MicroNet(**kwargs)
