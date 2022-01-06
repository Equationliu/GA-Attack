import timm
import dill
import torch
import torch.nn as nn

import torchvision

import sys
sys.path.append("..")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

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

def get_architecture(model_name="resnet152") -> torch.nn.Module:

    #################
    # Training Models
    #################
    if model_name == "deit_base_distilled_patch16_384":
        model = timm.create_model('deit_base_distilled_patch16_384', pretrained=True)
        model.input_size = 384
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "dm_nfnet_f1":
        model = timm.create_model("dm_nfnet_f1", pretrained=True)
        model.input_size = 320
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "tf_efficientnet_b4_ns":
        model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        model.input_size = 380
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "tf_efficientnet_b5_ns":
        model = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "ecaresnet269d":
        model = timm.create_model('ecaresnet269d', pretrained=True)
        model.input_size = 352
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "resnet_v2_152":
        model = timm.create_model('resnetv2_152x2_bitm', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "ig_resnext101_32x16d":
        model = timm.create_model('ig_resnext101_32x16d', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "InceptionV4":
        model = timm.create_model('inception_v4', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)


    ###################
    # Validation Models
    ###################
    elif model_name == "resnet_v2_50":
        model = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "resnet_v2_101":
        model = timm.create_model('resnetv2_101x1_bitm', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "InceptionV3":
        model = timm.create_model('tf_inception_v3', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "InceptionResnetV2":
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        model.input_size = 299
        normalize_layer = NormalizeLayer(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    elif model_name == "cspdarknet53":
        model = timm.create_model('cspdarknet53', pretrained=True)
        model.input_size = 256
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "densenet201":
        model = timm.create_model('densenet201', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    elif model_name == "repvgg_b2g4":
        model = timm.create_model('repvgg_b2g4', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        
    elif model_name == "dpn107":
        model = timm.create_model('dpn107', pretrained=True)
        model.input_size = 224
        normalize_layer = NormalizeLayer(mean=IMAGENET_DPN_MEAN, std=IMAGENET_DPN_STD)
        
    else:
        raise Exception("Not Supported Model Name!")
        
    return torch.nn.Sequential(normalize_layer, model)