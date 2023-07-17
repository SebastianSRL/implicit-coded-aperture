import torch
import numpy as np
from torch.autograd import Function

def dims2coords(dims):
    idx = range(len(dims) + 1)
    dims = [np.linspace(-1, 1, d) for d in dims]
    coords = np.array(np.meshgrid(*dims, indexing='ij'))
    coords = np.transpose(coords, tuple(idx[1:]) + (idx[0],))
    return coords

def normalize(x, low=0, high=1):
    MAX = torch.max(x)
    MIN = torch.min(x)
    return (high - low) * (x - MIN) / (MAX - MIN) + low

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# main functions
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

def save_config(save_path, file, args):
    txt_name = "/model_info.txt"

    with open(save_path + txt_name, 'w') as txt:
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write('             Model information             \n')
        txt.write('#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\n')
        txt.write(f'python {file} \\\n')
        for arg in vars(args):
            name = arg.replace('_', '-')
            value = getattr(args, arg)

            if isinstance(value, dict):
                value = f'"{value}"'
            txt.write(f'       --{name} {value} \\\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
