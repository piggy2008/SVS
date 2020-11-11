import torch
import torch.nn as nn

class Alternate(nn.Module):
    def __init__(self, division=8):
        super(Alternate, self).__init__()
        self.fuse_conv = nn.Conv2d(64*2, 64, kernel_size=3, padding=1)
        self.fuse_bn = nn.BatchNorm2d(64)
        self.division = division

    def initialize(self):
        nn.init.kaiming_normal_(self.fuse_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.ones_(self.fuse_bn.weight)
        nn.init.zeros_(self.fuse_bn.bias)

    def forward(self, x, flow):
        x_split = torch.split(x, self.division, dim=1)
        flow_split = torch.split(flow, self.division, dim=1)
        feat_cat = []
        for i, j in zip(x_split, flow_split):
            feat_cat.append(i)
            feat_cat.append(j)
        feat_cat = torch.cat(feat_cat, dim=1)
        feat = torch.relu(self.fuse_bn(self.fuse_conv(feat_cat)))
        return feat

if __name__ == '__main__':
    input = torch.zeros([2, 64, 20, 20])
    net = Alternate()
    output = net(input, input)
    print(output.size())


