import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Ensemble_logits(nn.Module):

    def __init__(self, model_list, Num_classes=1000, prob=0.7, mode="nearest"):
        super(Ensemble_logits, self).__init__()
        self.model_list = model_list
        self.length = len(self.model_list)
        self.classes = Num_classes
        self.prob = prob
        self.mode = mode

    def resize(self, input, target_size):
        return F.interpolate(input, size=target_size, mode=self.mode)

    def input_diversity(self, input_tensor, input_size, target_scale=0.1):
        target_size = int(input_size * (target_scale + 1.0))
        small_size = int(input_size * (1.0 -target_scale))
        rnd = np.floor(np.random.uniform(small_size, target_size, size=())).astype(np.int32).item()
        x_resize = self.resize(input_tensor, rnd)
        h_rem = target_size - rnd
        w_rem = target_size - rnd
        pad_top = np.floor(np.random.uniform(0, h_rem, size=())).astype(np.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = np.floor(np.random.uniform(0, w_rem, size=())).astype(np.int32).item()
        pad_right = w_rem - pad_left
        padded = F.pad(x_resize, (int(pad_top), int(pad_bottom),int(pad_left), int(pad_right), 0, 0, 0, 0))
        if torch.rand(1) <= self.prob:
            return self.resize(padded, input_size)
        else:
            return self.resize(input_tensor, input_size)

    def forward(self, x, diversity=False):
        if diversity:
            output = torch.cat([self.model_list[idx](self.input_diversity(x, input_size=self.model_list[idx][1].input_size)).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)
        else:
            output = torch.cat([self.model_list[idx](self.resize(x, target_size=self.model_list[idx][1].input_size)).unsqueeze(1) for idx in range(self.length)], dim = 1)
            return output.mean(dim = 1)