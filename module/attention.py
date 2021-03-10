import torch.nn as nn
import torch
import torch.nn.functional as F
import math

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
        else:
            m.initialize()

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim * 3, out_channels=in_dim, kernel_size=1)

    def initialize(self):
        weight_init(self)

    def forward(self, low, high, flow):

        batch, channel, height, width = low.size()

        combined = torch.cat([low, high, flow], dim=1)

        proj_query = self.query_conv(combined).view(batch, channel, -1)
        proj_key = self.key_conv(combined).view(batch, channel, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy = ((self.chanel_in) ** -.5) * energy
        attention = F.softmax(energy)
        proj_value = self.value_conv(combined).view(batch, channel, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch, channel, height, width)
        return out

if __name__ == '__main__':
    x = torch.rand([2, 64, 70, 70])
    y = torch.rand([2, 64, 70, 70])
    z = torch.rand([2, 64, 70, 70])
    model = CAM_Module(in_dim=64)
    out = model(x, y, z)
    print(out.size())