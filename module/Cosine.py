import torch
import torch.nn as nn
import torch.nn.functional as F

class Cosine(nn.Module):
    def __init__(self):
        super(Cosine, self).__init__()

    def initialize(self):
        pass

    def forward(self, low, high, flow_low, flow_high):
        squeeze_array = []
        for tensor in [low, high, flow_low, flow_high]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))

        cos_lh = torch.div(torch.sum(squeeze_array[0] * squeeze_array[1], dim=1, keepdim=True),
                  torch.norm(squeeze_array[0], 2, 1).view(-1, 1)
                  * torch.norm(squeeze_array[1], 2, 1).view(-1, 1) + 1e-7)

        cos_lfl = torch.div(torch.sum(squeeze_array[0] * squeeze_array[2], dim=1, keepdim=True),
                  torch.norm(squeeze_array[0], 2, 1).view(-1, 1)
                  * torch.norm(squeeze_array[2], 2, 1).view(-1, 1) + 1e-7)

        cos_hfh = torch.div(torch.sum(squeeze_array[1] * squeeze_array[3], dim=1, keepdim=True),
                  torch.norm(squeeze_array[1], 2, 1).view(-1, 1)
                  * torch.norm(squeeze_array[3], 2, 1).view(-1, 1) + 1e-7)

        cos_flfh = torch.div(torch.sum(squeeze_array[2] * squeeze_array[3], dim=1, keepdim=True),
                  torch.norm(squeeze_array[2], 2, 1).view(-1, 1)
                  * torch.norm(squeeze_array[3], 2, 1).view(-1, 1) + 1e-7)

        weight_lh = torch.exp(cos_lh) / (torch.exp(cos_lh) + torch.exp(cos_lfl) + 1e-7)
        weight_lfl = 1 - weight_lh
        weight_hl = torch.exp(cos_lh) / (torch.exp(cos_lh) + torch.exp(cos_hfh) + 1e-7)
        weight_hfh = 1 - weight_hl
        weigh_fll = torch.exp(cos_lfl) / (torch.exp(cos_lfl) + torch.exp(cos_flfh) + 1e-7)
        weight_flfh = 1 - weigh_fll
        weight_fhfl = torch.exp(cos_flfh) / (torch.exp(cos_flfh) + torch.exp(cos_hfh) + 1e-7)
        weight_fhh = 1 - weight_fhfl

        weight_lh = weight_lh.view(weight_lh.shape + (1,) * 2)
        weight_lfl = weight_lfl.view(weight_lfl.shape + (1,) * 2)
        weight_hl = weight_hl.view(weight_hl.shape + (1,) * 2)
        weight_hfh = weight_hfh.view(weight_hfh.shape + (1,) * 2)
        weight_fll = weigh_fll.view(weigh_fll.shape + (1,) * 2)
        weight_flfh = weight_flfh.view(weight_flfh.shape + (1,) * 2)
        weight_fhfl = weight_fhfl.view(weight_fhfl.shape + (1,) * 2)
        weight_fhh = weight_fhh.view(weight_fhh.shape + (1,) * 2)

        message_l = weight_lh * high + weight_lfl * flow_low
        message_h = weight_hl * low + weight_hfh * flow_high
        message_fh = weight_fhh * high + weight_fhfl * flow_low
        message_fl = weight_flfh * flow_high + weight_fll * low

        return message_l, message_h, message_fl, message_fh

if __name__ == '__main__':
    net = Cosine()
    input = torch.zeros([2, 64, 20, 20])

    out = net(input, input, input, input)