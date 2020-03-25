import torch
from torch import nn as nn
from torch.optim.optimizer import Optimizer
import math
import matplotlib.pylab as plt
import numpy as np
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self,pred, target, weights=None, ignore_index=0):
        """
        output : NxCxxDxHxW Variable
        target :  NxCxxDxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
        pred = pred.float()
        # pred_max = pred.data.max(1)[1]
        # one_hot_pred = to_one_hot(pred_max, target.shape[1]).cuda(0).float()
        # one_hot_pred.requires_grad=True
        # pred = one_hot_pred
        pred[pred >= 0.5] = 1.
        pred[pred < 0.5] = 0.
        target = target.float()
        eps = 1e-8
        dims = (2, 3,4)
        intersection = torch.sum((pred * target), dims)
        cardinality = torch.sum((pred), dims) + torch.sum((target), dims)
        dice_score = (2. * intersection + eps) / (cardinality + eps)
        index = torch.mean(torch.tensor(1.) - dice_score)
        return index


class Adam16(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
        # for p in group['params']:

        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group['params'], fp32_group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

def plot(train, val, epoch):
    plt.plot(np.arange(len(train)), train, c='red', label='Training loss')
    plt.plot(np.arange(len(val)), val, c='blue', label='Validation loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('./Loss Curve')
    plt.show()

def to_one_hot(maxes, num_classes):
    one_hot = F.one_hot(maxes, num_classes)
    return one_hot.permute(0, 4, 1, 2, 3)