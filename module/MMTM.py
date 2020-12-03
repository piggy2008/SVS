import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from module.GCN import GCN
import math

coarse_adj_list = [
            # 1  2  3
            [0.333, 0.333, 0.333],  # 1
            [0.333, 0.333, 0.333],  # 2
            [0.333, 0.333, 0.333],  # 3
        ]

coarse_adj_list2 = [
            # 1  2  3  4
            [0.25, 0.25, 0.25, 0.25],  # 1
            [0.25, 0.25, 0.25, 0.25],  # 2
            [0.25, 0.25, 0.25, 0.25],  # 3
            [0.25, 0.25, 0.25, 0.25],  # 4
        ]

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
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
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
        elif isinstance(m, GCN):
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
        self.fc_two = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_b),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
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
        excitation2 = self.fc_two(combined_fc).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(combined_fc).view(batch, channel, 1, 1)

        weighted_feat_a = a + excitation2 * b + excitation3 * c
        weighted_feat_b = b + excitation1 * a + excitation3 * c
        weighted_feat_c = c + excitation1 * a + excitation2 * b

        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c], dim=1)
        atten_a = self.gate_a(feat_cat)
        atten_b = self.gate_b(feat_cat)
        atten_c = self.gate_c(feat_cat)

        attention_vector = torch.cat([atten_a, atten_b, atten_c], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_a, attention_vector_b, attention_vector_c = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :], attention_vector[:, 2:3, :, :]

        merge_feature = a * attention_vector_a + b * attention_vector_b + c * attention_vector_c
        out_a = torch.relu((a + merge_feature) / 2)
        out_b = torch.relu((b + merge_feature) / 2)
        out_c = torch.relu((c + merge_feature) / 2)
        return out_a, out_b, out_c, merge_feature

class SETriplet2(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_out):
        super(SETriplet2, self).__init__()
        dim = dim_a + dim_b + dim_c

        self.gcn = GCN(3, 64, 64)
        self.adj = torch.from_numpy(np.array(coarse_adj_list)).float()

        self.fc_one = nn.Sequential(
            nn.Linear(dim, dim_a),
            nn.Sigmoid()
        )
        self.fc_two = nn.Sequential(
            nn.Linear(dim, dim_b),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
            nn.Linear(dim, dim_c),
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
        combined_fc = self.avg_pool(combined).view(batch, 3, channel)
        batch_adj = self.adj.repeat(batch, 1, 1)
        batch_adj = batch_adj.cuda()
        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation1 = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel, 1, 1)

        weighted_feat_a = a + excitation2 * b + excitation3 * c
        weighted_feat_b = b + excitation1 * a + excitation3 * c
        weighted_feat_c = c + excitation1 * a + excitation2 * b

        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c], dim=1)
        atten_a = self.gate_a(feat_cat)
        atten_b = self.gate_b(feat_cat)
        atten_c = self.gate_c(feat_cat)

        attention_vector = torch.cat([atten_a, atten_b, atten_c], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_a, attention_vector_b, attention_vector_c = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :], attention_vector[:, 2:3, :, :]

        merge_feature = a * attention_vector_a + b * attention_vector_b + c * attention_vector_c
        out_a = torch.relu((a + merge_feature) / 2)
        out_b = torch.relu((b + merge_feature) / 2)
        out_c = torch.relu((c + merge_feature) / 2)
        return out_a, out_b, out_c, merge_feature

class SEQuart(nn.Module):
    def __init__(self, dim_a, dim_b, dim_c, dim_d):
        super(SEQuart, self).__init__()
        dim = dim_a + dim_b + dim_c + dim_d

        self.gcn = GCN(4, 64, 64)
        self.adj = torch.from_numpy(np.array(coarse_adj_list2)).float()

        self.fc_one = nn.Sequential(
            nn.Linear(dim, dim_a),
            nn.Sigmoid()
        )
        self.fc_two = nn.Sequential(
            nn.Linear(dim, dim_b),
            nn.Sigmoid()
        )
        self.fc_three = nn.Sequential(
            nn.Linear(dim, dim_c),
            nn.Sigmoid()
        )
        self.fc_four = nn.Sequential(
            nn.Linear(dim, dim_c),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        # self.gate_a = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_b = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_c = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # self.gate_d = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim, 4, kernel_size=1, bias=True)

    def initialize(self):
        weight_init(self)

    def forward(self, low, high, flow, feedback):
        batch, channel, _, _ = low.size()
        combined = torch.cat([low, high, flow, feedback], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, 4, channel)
        batch_adj = self.adj.repeat(batch, 1, 1)
        batch_adj = batch_adj.cuda()
        feat_mean, feat_cat = self.gcn(combined_fc, batch_adj)

        excitation1 = self.fc_one(feat_cat).view(batch, channel, 1, 1)
        excitation2 = self.fc_two(feat_cat).view(batch, channel, 1, 1)
        excitation3 = self.fc_three(feat_cat).view(batch, channel, 1, 1)
        excitation4 = self.fc_four(feat_cat).view(batch, channel, 1, 1)

        weighted_feat_a = low + excitation2 * high + excitation3 * flow + excitation4 * feedback
        weighted_feat_b = excitation1 * low + high + excitation3 * flow + excitation4 * feedback
        weighted_feat_c = excitation1 * low + excitation2 * high + flow + excitation4 * feedback
        weighted_feat_d = excitation1 * low + excitation2 * high + excitation3 * flow + feedback

        feat_cat = torch.cat([weighted_feat_a, weighted_feat_b, weighted_feat_c, weighted_feat_d], dim=1)
        # atten_a = self.gate_a(feat_cat)
        # atten_b = self.gate_b(feat_cat)
        # atten_c = self.gate_c(feat_cat)
        # atten_d = self.gate_d(feat_cat)
        # attention_vector = torch.cat([atten_a, atten_b, atten_c, atten_d], dim=1)
        attention_vector = self.gate(feat_cat)
        attention_vector = self.softmax(attention_vector)

        attention_vector_a, attention_vector_b = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        attention_vector_c, attention_vector_d = attention_vector[:, 2:3, :, :], attention_vector[:, 3:4, :, :]
        merge_feature = low * attention_vector_a + high * attention_vector_b + \
                        flow * attention_vector_c + feedback * attention_vector_d
        # bug backup
        # merge_feature = low * attention_vector_a + high * attention_vector_b + \
        #                 flow * attention_vector_c * feedback * attention_vector_d

        return merge_feature

if __name__ == '__main__':
        input = torch.zeros([2, 64, 24, 24])
        net = SEQuart(64, 64, 64, 64)
        output = net(input, input, input, input)

