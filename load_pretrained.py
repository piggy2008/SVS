import torch
from models.net import SNet
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
    flow_keys = [key for key in mga_keys if key.find('resnet_aspp.backbone_features') > -1]
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

if __name__ == '__main__':
    net = SNet(cfg=None).cuda()
    net = fuse_MGA_F3Net('../pre-trained/MGA_trained.pth', '../pre-trained/F3Net.pth', net)
    torch.save(net.state_dict(), '../pre-trained/SNet.pth')
