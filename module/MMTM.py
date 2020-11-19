import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

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
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        else:
            m.initialize()

class MMTM(nn.Module):
    def __init__(self, dim_a, dim_b, ratio):
        super(MMTM, self).__init__()
        dim = dim_a + dim_b
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_a = nn.Linear(dim_out, dim_a)
        self.fc_b = nn.Linear(dim_out, dim_b)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def initialize(self):
        nn.init.kaiming_normal_(self.fc_squeeze.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_a.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_b.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, a, b):
        squeeze_array = []
        for tensor in [a, b]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_a(excitation)
        sk_out = self.fc_b(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(a.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(b.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return a * vis_out, b * sk_out

class SETriplet(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_out):
        super(SETriplet, self).__init__()
        dim = dim_a + dim_b + dim_c
        # self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_one = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_a),
            nn.Sigmoid()
        )
        self.fc_anthoer = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_b),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, a, b, c):
        batch, channel, _, _ = a.size()
        combined = torch.cat([a, b, c], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, channel * 3)
        excitation1 = self.fc_one(combined_fc).view(batch, channel, 1, 1)
        excitation2 = self.fc_anthoer(combined_fc).view(batch, channel, 1, 1)

        weighted_feat_a = a + excitation1 * b + excitation2 * c
        weighted_feat_b = b + excitation1 * a + excitation2 * c
        weighted_feat_c = c + excitation1 * a + excitation2 * b

        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c], dim=1)
        atten_a = self.gate_a(feat_cat)
        atten_b = self.gate_b(feat_cat)
        atten_c = self.gate_c(feat_cat)

        attention_vector = torch.cat([atten_a, atten_b, atten_c], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_a, attention_vector_b, attention_vector_c = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :], attention_vector[:, 2:3, :, :]

        merge_feature = a * attention_vector_a + b * attention_vector_b + c * attention_vector_c
        return merge_feature

if __name__ == '__main__':
        input = torch.zeros([2, 64, 24, 24])
        net = SETriplet(64, 64, 64, 64)
        output = net(input, input, input)

