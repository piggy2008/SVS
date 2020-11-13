import torch
import torch.nn as nn
import torch.nn.functional as F
# from matplotlib import pyplot as plt

class EP(nn.Module):
    def __init__(self):
        super(EP, self).__init__()
        weight = nn.Parameter()

    def initialize(self):
        pass

    def forward(self, x, sal):
        b, c, w, h = x.size()
        fv = x.view(b, c, -1).permute(0, 2, 1)
        salv = sal.view(b, c, -1)
        fv = torch.bmm(fv, salv)
        p = F.softmax(fv, dim=-1)
        logp = F.log_softmax(fv, dim=-1)
        hp = - torch.sum(p * logp, dim=-1)
        values, index = torch.max(hp, dim=-1, keepdim=True)
        pw = 1 - hp / values
        # print(hp.size())
        pa_map = pw.view(b, -1, w, h)
        # print('x:', torch.mean(x))
        # print('pa_map * x:', torch.mean(pa_map * x))
        return pa_map * x

if __name__ == '__main__':
    input = torch.randn([2, 64, 20, 20])
    sal = torch.randn([2, 64, 10, 10])
    net = EP()
    output = net(input, sal)
    print(torch.unique(output))
    # output = output.data.cpu().numpy()
    # from matplotlib import pyplot as plt
    #
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(output[0, 0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(output[1, 0])
    # plt.show()
    # print(output.size())


