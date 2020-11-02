#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from MGA.ResNet import ResNet34
from module.ConGRUCell import ConvGRUCell

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('pre-trained/resnet50-19c8e357.pth'), strict=False)

class SFM(nn.Module):
    def __init__(self, GNN=False):
        super(SFM, self).__init__()
        self.conv1h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.GNN = GNN
        if self.GNN:
            self.conGRU = ConvGRUCell(64, 64, 1)
            self.iterate_time = 5

    def forward(self, low, high, flow):
        if high.size()[2:] != low.size()[2:]:
            high = F.interpolate(high, size=low.size()[2:], mode='bilinear')
        if flow.size()[2:] != low.size()[2:]:
            flow = F.interpolate(flow, size=low.size()[2:], mode='bilinear')
        out1h = self.conv1h(high)
        out2h = self.conv2h(out1h)
        out1l = self.conv1l(low)
        out2l = self.conv2l(out1l)
        out1f = self.conv1f(flow)
        out2f = self.conv2f(out1f)
        if self.GNN:
            for passing in range(self.iterate_time):
                message_h = out2l + out2f
                message_l = out2h + out2f
                msssage_f = out2h + out2l

                h_h = self.conGRU(message_h, out2h)
                h_l = self.conGRU(message_l, out2l)
                h_f = self.conGRU(msssage_f, out2f)

                out2h = h_h.clone()
                out2l = h_l.clone()
                out2f = h_f.clone()

                if passing == self.iterate_time - 1:
                    fuse = out2h * out2l * out2f
        else:
            fuse = out2h * out2l * out2f

        out3h = self.conv3h(fuse) + out1h
        out4h = self.conv4h(out3h)
        out3l = self.conv3l(fuse) + out1l
        out4l = self.conv4l(out3l)
        out3f = self.conv3f(fuse) + out1f
        out4f = self.conv4f(out3f)

        return out4l, out4h, out4f

    def initialize(self):
        weight_init(self)

class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)

class Decoder_flow(nn.Module):
    def __init__(self, GNN=False):
        super(Decoder_flow, self).__init__()
        self.cfm45  = SFM(GNN=GNN)
        self.cfm34  = SFM(GNN=GNN)
        self.cfm23  = SFM(GNN=False)

    def forward(self, out2h, out3h, out4h, out5v, out2f, out3f, out4f, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v, out4b = self.cfm45(out4h + refine4, out5v, out4f + refine4)
            out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h + refine3, out4f, out3f + out4b + refine3)
            out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h+refine2, out3v, out2f + out3b + refine2)
        else:
            out4h, out4v, out4b = self.cfm45(out4h, out5v, out4f)
            out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h, out4v, out3f + out4b)
            out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h, out3v, out2f + out3b)
        return out2h, out3h, out4h, out5v, out2b, out3b, out4b, pred

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v)
            out3h, out3v = self.cfm34(out3h+refine3, out4v)
            out2h, pred  = self.cfm23(out2h+refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred  = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


class SNet(nn.Module):
    def __init__(self, cfg, GNN=False):
        super(SNet, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.flow_bkbone = ResNet34(nInputChannels=3, os=16, pretrained=False)
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.flow_align4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.flow_align1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder1 = Decoder_flow(GNN=GNN)
        self.decoder2 = Decoder_flow(GNN=GNN)
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearf2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearf3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearf4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x, flow, shape=None):
        out2h, out3h, out4h, out5v = self.bkbone(x) # layer1, layer2, layer3, layer4
        flow_layer4, flow_layer1, _, flow_layer2, flow_layer3 = self.flow_bkbone(flow)
        # print('flow size:', flow_layer4.size(), '--- image size:', out5v.size())
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        out2f, out3f, out4f = self.flow_align2(flow_layer2), self.flow_align3(flow_layer3), self.flow_align4(flow_layer4)
        out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1 = self.decoder1(out2h, out3h, out4h, out5v, out2f, out3f, out4f)
        out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2 = self.decoder2(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1)

        shape = x.size()[2:] if shape is None else shape
        pred1a = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2a = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        out2f = F.interpolate(self.linearr2(out2f), size=shape, mode='bilinear')
        out3f = F.interpolate(self.linearr3(out3f), size=shape, mode='bilinear')
        out4f = F.interpolate(self.linearr4(out4f), size=shape, mode='bilinear')

        return pred1a, pred2a, out2h, out3h, out4h, out5h, out2f, out3f, out4f



    def initialize(self):
        # if self.cfg.snapshot:
        #     self.load_state_dict(torch.load(self.cfg.snapshot))
        # else:
        weight_init(self)
