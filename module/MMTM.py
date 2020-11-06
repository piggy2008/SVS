import torch
import torch.nn as nn
import torch.nn.functional as F

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
