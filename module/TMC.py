import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net import weight_init
# def weight_init(module):
#     for n, m in module.named_children():
#         print('initialize: '+n)
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
#             nn.init.ones_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Sequential):
#             weight_init(m)
#         elif isinstance(m, nn.ReLU):
#             pass
#         else:
#             m.initialize()

class TMC(nn.Module):
    def __init__(self):
        super(TMC, self).__init__()
        self.channel = nn.Conv2d(64, 64, 1, bias=True)
        self.spatial = nn.Conv2d(64, 1, 1, bias=True)


    def initialize(self):
        nn.init.kaiming_normal_(self.channel.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.spatial.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, value, query):
        query_feat = self.spatial(query)
        query_feat = nn.Sigmoid()(query_feat)

        value_spatial_feat = query_feat * value

        # channel-wise attention
        feat_vec = F.adaptive_avg_pool2d(value_spatial_feat, (1, 1))
        feat_vec = self.channel(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        value_weighted_feat = value_spatial_feat * feat_vec

        final_feat = value_weighted_feat + value
        return final_feat