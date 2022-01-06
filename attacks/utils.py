import torch
import torch.nn.functional as F

import numpy as np
import scipy.stats as st

#############
# utils
#############

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_kernel(kernel_size=7):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3).transpose(2, 3, 0, 1)
    stack_kernel = torch.from_numpy(stack_kernel).cuda()
    return stack_kernel



########################################

# Encoder and Decoder for Featrue Attack
# Xu et al. Towards Feature Space Adversarial Attack
# https://github.com/qiulingxu/FeatureSpaceAttack#Pretrained-Model

########################################

class Encoder(object):

    def __init__(self, weights_path = "data/ckpts/vgg19_normalised.npz"):
        # load weights (kernel and bias) from npz file
        weights = np.load(weights_path)
        idx = 0
        self.weight_vars = []
        self.ENCODER_LAYERS = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1')

        for layer in self.ENCODER_LAYERS:
            kind = layer[:4]

            if kind == 'conv':
                kernel = weights['arr_%d' % idx]
                bias   = weights['arr_%d' % (idx + 1)]
                kernel = torch.from_numpy(kernel.astype(np.float32))
                bias   = torch.from_numpy(bias.astype(np.float32))
                idx += 2
                self.weight_vars.append((kernel, bias))

        IMAGENET_DEFAULT_MEAN = (103.939, 116.779, 123.68)
        self.mean_torch = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)

    def encode(self, image):
        idx = 0
        device = image.device
        layers = {}
        current = image
        for i,layer in enumerate(self.ENCODER_LAYERS):
            kind = layer[:4]

            if kind == 'conv':
                kernel, bias = self.weight_vars[idx]
                idx += 1
                current = conv2d(current, kernel.to(device), bias.to(device))

            elif kind == 'relu':
                current = F.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current
        
            # print("encoder {} shape of {}: ".format(i, layer), current.shape)

        assert(len(layers) == len(self.ENCODER_LAYERS))

        enc = layers[self.ENCODER_LAYERS[-1]]

        return enc, layers

    def preprocess(self, image, mode='RGB'):
        assert mode == "RGB"
        IMAGE_SHAPE = image.shape[2]
        # preprocess
        if IMAGE_SHAPE != 224:
            image = F.interpolate(image, size=224, mode="bilinear")
        
        # To BGR
        permute_RGBtoBGR = [2, 1, 0]
        image = image[:, permute_RGBtoBGR, :, :]

        # normalize
        return image - self.mean_torch.to(image.device)

    def deprocess(self, image, mode='BGR'):
        assert mode == "BGR"
        IMAGE_SHAPE = image.shape[2]
        image = image + self.mean_torch.to(image.device)
        
        permute_BGRtoRGB = [2, 1, 0]
        image = image[:, permute_BGRtoRGB, :, :]
        image = torch.clamp(image, 0.0, 255.0)
        
        return image


class Decoder(object):

    def __init__(self, weights_path="data/ckpts/imagenetshallowest.npy"):
        self.weight_vars = np.load(weights_path, allow_pickle = True).item()
        self.DECODER_LAYERS = ('conv2_1', 'conv1_2', 'conv1_1')
        self.upsample_indices = (0, )
        self.func = transconv2d
        self.final_layer_idx  = len(self.DECODER_LAYERS) - 1

    def decode(self, image):
        out = image

        for i, layer in enumerate(self.DECODER_LAYERS):
            kernel, bias = self.weight_vars['decoder/' + layer + '/kernel'], self.weight_vars['decoder/' + layer + '/bias']
            kernel = kernel.transpose(3, 2, 0, 1)
            kernel, bias = torch.from_numpy(kernel.astype(np.float32)).to(image.device), torch.from_numpy(bias.astype(np.float32)).to(image.device)

            if i == self.final_layer_idx:
                out = self.func(out, kernel, bias, use_relu=False)
            else:
                out = self.func(out, kernel, bias)

            if i in self.upsample_indices:
                out = upsample(out)

            # print("decoder out %d shape: "%i, out.shape)

        return out


def transconv2d(x, kernel, bias, use_relu=True, stride=1):
    bs = x.shape[0]
    img_sz = x.shape[1]
    filter_size = kernel.shape[2]
    # conv and add bias
    out = F.conv_transpose2d(x, kernel, bias=bias, padding=1, output_padding=0)
    if use_relu:
        out = F.relu(out)
    return out

def upsample(x, scale=2):
    height = x.shape[2]*scale
    width = x.shape[3]*scale
    output = F.interpolate(x, size=[height, width], mode="nearest")
    return output

def pool2d(x):
    return F.max_pool2d(x, kernel_size=(2, 2), stride=(2,2))

def conv2d(x, kernel, bias):
    x_padded = torch.nn.ReflectionPad2d(1)(x)
    # conv and add bias
    out = F.conv2d(x_padded, kernel, bias=bias, stride=1)
    return out

