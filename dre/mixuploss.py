import torch
import torch.nn as nn
import torch.nn.functional as F
# from captum.attr import LRP
from captum.attr import LayerGradCam


def norm(x):
    max_values = x.reshape(x.shape[0],-1).max(dim=-1,keepdim=True)[0]
    min_values = x.reshape(x.shape[0],-1).min(dim=-1,keepdim=True)[0]
    return (x - min_values.unsqueeze(2).unsqueeze(3)) / (max_values.unsqueeze(2).unsqueeze(3) - min_values.unsqueeze(2).unsqueeze(3))

def MixupLoss(inputs, targets, net, num_classes, device):
    exp_loss = 0.0
    sparse_loss = 0.0
    mix_ce_loss = 0.0
    alpha = 0.2
    beta = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    criterion = nn.L1Loss()
    layer_gc = LayerGradCam(net, net.module.network.layer4)

    for i in range(num_classes):
        if (targets == i).sum() == 0:
            continue
        inp = inputs[targets == i]
        tag = targets[targets == i]
        indexes = torch.randperm(inp.shape[0])
        inp_ = inp[indexes]

        attr = layer_gc.attribute(inp, target=i)
        attr_norm = norm(attr)
        attr_norm_ = attr_norm[indexes]

        mix_lambda = beta.rsample(sample_shape=(len(inp),1,1)).to(device)
        mixup = mix_lambda*inp + (1 - mix_lambda)*inp_
        attr_mixup = mix_lambda*attr_norm + (1 - mix_lambda)*attr_norm_

        attr_mix = layer_gc.attribute(mixup, target=i)
        attr_mix_norm = norm(attr_mix)

        exp_loss += criterion(attr_mix_norm, attr_mixup)
        sparse_loss += (torch.mean(torch.linalg.norm(attr_norm, dim=(1,2), ord=1)) + torch.mean(torch.linalg.norm(attr_mix_norm, dim=(1,2), ord=1))) / 2
        mix_ce_loss += F.cross_entropy(net(mixup), tag)
    return exp_loss/num_classes, sparse_loss/num_classes, mix_ce_loss/num_classes