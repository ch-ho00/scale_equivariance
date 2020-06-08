import torch
import torch.nn.functional as F
import torch.nn as nn
# TODO: try other loss functions
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
# MSE loss MultiLabelMarginLoss
 
class CrossEntropyLoss2d(nn.Module):
    '''
    Input
        inputs =(N,C) where C = number of classes  or (N,C,d1,d2 ... ,dk) in case of 2d Loss
        targets = (N) or (N, d1,d2 ... dk) where K > 1

        Note: for cityscapes dataset, C = 19

    Output = 
        negative log likelihood loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        # it is advised to exclude 255 in the class
        # narrow the scope down to 19 classes
        if targets.shape[-1] != inputs.shape[-1]:
            targets=targets[:,4:-4,4:-4]
        mask = torch.zeros_like(targets)
        mask = (targets < 19).float()

        inputs = torch.stack([inputs[:,i,:,:][mask==1]  for i in range(inputs.shape[1])]).permute(1,0)
        targets = targets[mask==1].long()

        return self.nll_loss(F.log_softmax(inputs), targets)



# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
