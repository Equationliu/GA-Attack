import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import scipy.stats as st

os.environ['TORCH_HOME']='~/.cache/torch/'

from utils import get_architecture
from ensemble_model import Ensemble_logits

parser = argparse.ArgumentParser('Running script', add_help=False)
parser.add_argument('--input_dir', default='./../input_dir', type=str)
parser.add_argument('--output_dir', default='./../output_dir', type=str)

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, meta_file, transform=None):

        self.data_dir = data_dir
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []

        with open(os.path.join(data_dir, meta_file)) as f:
            for line in f.readlines()[1:]:
                img_path, label = line.strip().split(',')
                self._indices.append((os.path.join(self.data_dir, 'images', img_path), label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        img_name = img_path.split('/')[-1]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_name

def tensor2img(input_tensor, save_dir, save_name):

    if input_tensor.is_cuda == True:
        input_tensor = input_tensor.cpu()

    input_tensor = input_tensor.permute(0, 2, 3, 1).data.numpy()
    for i in range(input_tensor.shape[0]):
        Image.fromarray((input_tensor[i] * 255).astype(np.uint8)).save('{}/{}'.format(save_dir, save_name[i]))
        print('{} saved in {}.'.format(save_name[i], save_dir))

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_kernel(kernel_size = 7):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3).transpose(2, 3, 0, 1)
    stack_kernel = torch.from_numpy(stack_kernel).cuda()
    return stack_kernel

def _get_norm_batch(x, p):
    return x.abs().pow(p).sum(dim=[1, 2, 3], keepdims=True).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return x / norm

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = ImageNetDataset(data_dir=args.input_dir, meta_file='dev.csv', transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print("Model to generate adv examples")
    # delete several models if OOM 
    model_name_list = ['deit_base_distilled_patch16_384', 'dm_nfnet_f1', 'tf_efficientnet_b4_ns',
                       'tf_efficientnet_b5_ns', 'ecaresnet269d', 'resnet_v2_152', 'ig_resnext101_32x16d', 'InceptionV4']
    print(model_name_list)
    model_list = []
    for model_name in model_name_list:
        temp_model = get_architecture(model_name=model_name).cuda()
        temp_model.eval()
        model_list.append(temp_model)

    # The Ensemble_logits Module use input diversity with 0.7 probability and nearest mode (interpolate) for each model
    ensemble_model = Ensemble_logits(
        model_list=model_list, Num_classes=1000, prob=0.7, mode="nearest").cuda()
    ensemble_model.eval()

    # eval model
    print("Model to predict adv examples's confidence")
    model_name_list = ["resnet_v2_50", "resnet_v2_101", "InceptionV3", "InceptionResnetV2", "cspdarknet53",
                       "densenet201", "repvgg_b2g4", "dpn107"]
    print(model_name_list)
    model_list = []
    for model_name in model_name_list:
        temp_model = get_architecture(model_name=model_name).cuda()
        temp_model.eval()
        model_list.append(temp_model)

    eval_model = Ensemble_logits(
        model_list=model_list, Num_classes=1000, prob=0.7, mode="nearest").cuda()
    eval_model.eval()

    for inputs, targets, img_name in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.shape[0]

        g = torch.zeros_like(inputs).cuda() # for momentum update
        delta = torch.zeros_like(inputs).cuda() # init perturbation
        mask = torch.ones((batch_size, )).bool()
        
        eps_list = [4.0 / 255, 6.0 / 255, 8.0 / 255, 12.0 / 255, 16.0 / 255, 32.0 / 255, 64.0 / 255] 
        for eps in eps_list:
            if eps <= 8.0 / 255:
                num_steps = 10
            elif eps <= 16.0 / 255:
                num_steps = 20
            else:
                num_steps = 50
            step_size = (eps * 1.25) / num_steps
            delta = Variable(delta.data, requires_grad=True)
            for _ in range(num_steps):
                delta.requires_grad_()
                adv = inputs + delta
                adv = torch.clamp(adv, 0, 1)
                with torch.enable_grad():
                    ensem_logits = ensemble_model(adv, diversity=True)
                    loss = F.cross_entropy(ensem_logits, targets, reduction="none")
                PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
                # gaussian filter with 5x5 kernel
                PGD_grad = F.conv2d(PGD_grad, weight=get_kernel(5),stride=(1, 1), groups=3, padding=(5 - 1) // 2)
                PGD_noise = normalize_by_pnorm(PGD_grad, p=1)
                g[mask] = g[mask] * 0.8 + PGD_noise[mask]
                delta = Variable(delta.data + step_size * torch.sign(g), requires_grad=True)
                delta = Variable(torch.clamp(delta.data, -eps, eps), requires_grad=True)

            # reset the memoried grad to zero when restart with a bigger eps
            g *= 0.0

            with torch.no_grad():
                tmp = inputs + delta
                tmp = torch.clamp(tmp, 0, 1)
                output = eval_model(tmp, diversity=False).data

            prob = F.softmax(output, dim = 1)
            conf = prob[np.arange(batch_size), targets.long()]
            # if the transfer confidence is still bigger than 1%, it may need bigger eps
            mask = (conf >= 0.01)

            if mask.sum() == 0:
                break

        print("Attack max eps level: {} finished, conf: {}".format(eps, conf))
        X_pgd = Variable(inputs + delta, requires_grad=False)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
        with torch.no_grad():
            ensem_logits = ensemble_model(X_pgd, diversity=False)
        out_X_pgd = ensem_logits.detach()
        err_pgd = (out_X_pgd.max(1)[1] != targets.data).float().sum()
        print("batch size: {}, attacked: {}".format(batch_size, err_pgd.item()))

        output_data = X_pgd.clone()
        tensor2img(input_tensor=output_data, save_dir=args.output_dir, save_name=img_name)
    
