import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Input_diversity(nn.Module):
    def __init__(self, model, args=None, num_classes=1000, prob=0.7, mode="nearest", diversity_scale=0.1):
        super(Input_diversity, self).__init__()
        self.model = model
        self.classes = num_classes
        self.prob = prob
        self.mode = mode
        self.diversity_scale = diversity_scale
        
        if args is not None:
            if not args.distributed or args.local_rank == 0:
                print("diversty prob: {}, resize mode: {}, diversity scale: {}".format(self.prob, self.mode, self.diversity_scale))

    def resize(self, input, target_size):
        return F.interpolate(input, size=target_size, mode=self.mode)

    def input_diversity(self, input_tensor, target_size):
        upper_bound = int(target_size * (self.diversity_scale + 1.0))
        lower_bound = int(target_size * (1.0 - self.diversity_scale))
        rnd = np.floor(np.random.uniform(lower_bound, upper_bound, size=())).astype(np.int32).item()
        x_resize = self.resize(input_tensor, rnd)
        h_rem = upper_bound - rnd
        w_rem = upper_bound - rnd
        pad_top = np.floor(np.random.uniform(0, h_rem, size=())).astype(np.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = np.floor(np.random.uniform(0, w_rem, size=())).astype(np.int32).item()
        pad_right = w_rem - pad_left
        padded = F.pad(x_resize, (int(pad_top), int(pad_bottom),int(pad_left), int(pad_right), 0, 0, 0, 0))
        if torch.rand(1) <= self.prob:
            return self.resize(padded, target_size)
        else:
            return self.resize(input_tensor, target_size)

    def forward(self, x, diversity=False):
        if diversity:
            return self.model(self.input_diversity(x, target_size=self.model[1].input_size))
        else:
            return self.model(self.resize(x, target_size=self.model[1].input_size))


class MultiEnsemble(Input_diversity):
    def __init__(self, model_list, **kwargs):
        super(MultiEnsemble, self).__init__(model=None, **kwargs)
        self.model_list = model_list
        self.length = len(self.model_list)

    def forward(self, x, diversity=False):
        if diversity:
            output = torch.cat([self.model_list[idx](self.input_diversity(x, target_size=self.model_list[idx][1].input_size)).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)
        else:
            output = torch.cat([self.model_list[idx](self.resize(x, target_size=self.model_list[idx][1].input_size)).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)
