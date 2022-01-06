import torch
import torchvision
import torch.nn as nn

import timm
from .Resnet import resnet152, resnet101_denoise, resnet152_denoise
from .Rvt import rvt_tiny

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self._mean_torch = torch.tensor(mean).view(1, 3, 1, 1).cuda()
        self._std_torch = torch.tensor(std).view(1, 3, 1, 1).cuda()

    def forward(self, inputs: torch.tensor):
        self._mean_torch = self._mean_torch.to(inputs.device)
        self._std_torch = self._std_torch.to(inputs.device)
        out = (inputs - self._mean_torch) / self._std_torch
        return out

class Denoise_NormalizeLayer(nn.Module):
    def __init__(self):
        super(Denoise_NormalizeLayer, self).__init__()

    def forward(self, inputs: torch.tensor):
        # RGB to BGR
        permute_RGBtoBGR = [2, 1, 0]
        inputs = inputs[:, permute_RGBtoBGR, :, :]
        # normalize
        out = (inputs / 0.5) - 1
        return out

def parse_dict(state_dict):
    return {k[7:]: v for k, v in state_dict.items()}





def get_architecture(model_name="InceptionV3") -> torch.nn.Module:

    ############################
    # Attacking models in Tab. 1 
    ############################

    ################################
    # Eight Naturally trained models
    ################################

    # Three Transformers
    if model_name == "vit_small_patch16_224":
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "swin_base_patch4_window7_224":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    # Three ResNets
    elif model_name == "swsl_resnext101_32x8d":
        model = timm.create_model('swsl_resnext101_32x8d', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "ssl_resnext50_32x4d":
        model = timm.create_model('ssl_resnext50_32x4d', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "swsl_resnet50":
        model = timm.create_model('swsl_resnet50', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    # Inception
    elif model_name == "InceptionV3":
        model = timm.create_model('tf_inception_v3', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "InceptionResnetV2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    ###########################################
    # Two Ensemble Adversarially trained models
    ###########################################

    elif model_name == "adv_inception_v3":
        model = timm.create_model('adv_inception_v3', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "ens_adv_inception_resnet_v2":
        model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)




    ##################################################################
    # Benchmarking models in Tab. 3 except ens_adv_inception_resnet_v2
    ##################################################################

    elif model_name == "Fast_AT":
        # Fast adversarial training using FGSM, ICLR 2020
        # https://github.com/locuslab/fast_adversarial/tree/master/ImageNet
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load("data/ckpts/imagenet_model_weights_4px.pth.tar", map_location='cpu') 
        model.load_state_dict(parse_dict(checkpoint['state_dict']))
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "Free_AT":
        # Adversarial Training for Free, NIPS 2019
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load("data/ckpts/model_best.pth.tar", map_location='cpu')
        model.load_state_dict(parse_dict(checkpoint['state_dict']))
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)


    # Feature Denoising for Improving Adversarial Robustness, CVPR 2019
    # for correct evaluation, it is better to evaluate feature denoising with tensorflow
    # https://github.com/facebookresearch/ImageNet-Adversarial-Training

    elif model_name == "AdvResnet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.load_state_dict(torch.load("data/ckpts/res152-adv.checkpoint"))
        model.input_size = 224
        normalize_layer = Denoise_NormalizeLayer()

    elif model_name == "Resnext101-DenoiseAll":
        model = resnet101_denoise()
        model.load_state_dict(torch.load("data/ckpts/Adv_Denoise_Resnext101.pytorch"))
        model.input_size = 224
        normalize_layer = Denoise_NormalizeLayer()

    elif model_name == "Resnet152-DenoiseAll":
        model = resnet152_denoise()
        model.load_state_dict(torch.load("data/ckpts/Adv_Denoise_Resnet152.pytorch"))
        model.input_size = 224
        normalize_layer = Denoise_NormalizeLayer()

    
    
    ## Other models

    elif model_name == "RVT-Tiny":
        # RVT: Towards Robust Vision Transformer
        # https://github.com/vtddggg/Robust-Vision-Transformer
        model = rvt_tiny(pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "DeepAugment_AugMix":
        # The many faces of robustness: A critical analysis of out-of-distribution generalization, ICCV 2021
        # https://github.com/hendrycks/imagenet-r
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load("data/ckpts/deepaugment_and_augmix.pth.tar", map_location='cpu')
        model.load_state_dict(parse_dict(checkpoint['state_dict']))
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "tf_efficientnet_l2_ns_475":
        model = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
        model.input_size = 475
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "swin_large_patch4_window12_384":
        model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
        model.input_size = 384
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    else:
        raise Exception("Not Supported Model Name!")
        
    return torch.nn.Sequential(normalize_layer, model)




    