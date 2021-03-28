import torch
from models.net import SNet
from models.net_i101 import INet101
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/tangyi/code/SVS')
plt.style.use('classic')

def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

def fuse_MGA_F3Net(mga_model_path, f3net_path, net, device_id=0):
    # net = SNet(cfg=None).cuda()
    f3_model = torch.load(f3net_path, map_location='cuda:' + str(device_id))
    mga_model = torch.load(mga_model_path, map_location='cuda:' + str(device_id))
    mga_keys = list(mga_model.keys())
    # flow_keys = [key for key in mga_keys if key.find('resnet_aspp.backbone_features') > -1]
    m_dict = net.state_dict()
    for k in m_dict.keys():
        if k in f3_model.keys():
            print('loading F3Net key:', k)
            param = f3_model.get(k)
            m_dict[k].data = param
        elif k.find('flow_bkbone') > -1:
            print('loading MGA key:', k)
            k_tmp = k.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            # k_tmp.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            m_dict[k].data = mga_model.get(k_tmp)
        else:
            print('not loading key:', k)

    net.load_state_dict(m_dict)
    return net

def fuse_MGA_F3Net2(mga_model_path, net, device_id=0):
    # net = SNet(cfg=None).cuda()
    # f3_model = torch.load(f3net_path, map_location='cuda:' + str(device_id))
    mga_model = torch.load(mga_model_path, map_location='cuda:' + str(device_id))
    mga_keys = list(mga_model.keys())
    # flow_keys = [key for key in mga_keys if key.find('resnet_aspp.backbone_features') > -1]
    m_dict = net.state_dict()
    for k in m_dict.keys():
        if k.find('flow_bkbone') > -1:
            print('loading MGA key:', k)
            k_tmp = k.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            # k_tmp.replace('flow_bkbone', 'resnet_aspp.backbone_features')
            m_dict[k].data = mga_model.get(k_tmp)
        elif k.find('bkbone.') > -1:
            print('loading MGA RGB backbone key:', k)
            k_tmp = k.replace('bkbone.', '')
            # print('loading MGA RGB backbone key:', k_tmp)
            m_dict[k].data = mga_model.get(k_tmp)
        else:
            print('not loading key:', k)

    net.load_state_dict(m_dict)
    return net

if __name__ == '__main__':
    net = INet101(cfg=None).cuda()
    # mga_model = torch.load('pre-trained/MGA_trained.pth')
    # mga_keys = list(mga_model.keys())
    # print(mga_keys)
    net = fuse_MGA_F3Net2('pre-trained/MGA_trained.pth', net, device_id=0)
    torch.save(net.state_dict(), 'pre-trained/SNet101.pth')
