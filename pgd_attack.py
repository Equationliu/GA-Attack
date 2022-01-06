import os
import time
import random
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

os.environ['TORCH_HOME']='~/.cache/torch/'

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# custom attack
from attacks.linf import PGD20

from utils import MyCustomDataset, get_architecture, Input_diversity
from utils import CrossEntropyLoss, MarginLoss

parser = argparse.ArgumentParser(description='PyTorch PGD Attack')
parser.add_argument('--batch_size', type=int, default=20,metavar='N', help='batch size for attack (default: 30)')
parser.add_argument('--mode', type=str, default="nearest")
parser.add_argument('--loss_fn', type=str, default="ce")
parser.add_argument('--epsilon', default=4.0, type=float,help='perturbation, (default: 4.0)')
parser.add_argument('--num_steps', default=20, type=int,help='number of steps')
parser.add_argument('--source_id', default=3, type=int,help='InceptionV3 as default')
parser.add_argument('--prob', default=0.7, type=float,help='input diversity prob')
parser.add_argument('--scale', default=0.1, type=float,help='input diversity scale')
parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

NUM_CLASSES = 1000

MODEL_NAME_DICT = {
    0: "ens_adv_inception_resnet_v2",
    1: "Fast_AT",
    2: "Free_AT",
    3: "AdvResnet152",
    4: "Resnext101-DenoiseAll",
    5: "Resnet152-DenoiseAll",
    6: "RVT-Tiny",
    7: "DeepAugment_AugMix",
    8: "tf_efficientnet_l2_ns_475",
    9: "swin_large_patch4_window12_384",
}

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main(args):

    init_seeds(cuda_deterministic=True)

    # Source model
    if not args.distributed or args.local_rank == 0:
        print("Using {} as source model!".format(MODEL_NAME_DICT[args.source_id]))
    tmp_model = get_architecture(model_name=MODEL_NAME_DICT[args.source_id]).cuda().eval()
    Source_model = Input_diversity(tmp_model, args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    
    loader = MyCustomDataset(img_path="data/images")
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(loader)
    else:
        sampler = torch.utils.data.SequentialSampler(loader)

    attack_loader = torch.utils.data.DataLoader(dataset=loader,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)

    natural_err_total, pgd_err_total = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    count = 0
    time_start = time.time()
    if not args.distributed or args.local_rank == 0:
        print("Starting time counting: {}\n".format(time_start))

    # loss_fn
    if args.loss_fn == "ce":
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == "margin":
        loss_fn = MarginLoss()
    else:
        raise Exception("invalid loss function!")


    for (img, label, img_name) in attack_loader:
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        with torch.no_grad():
            out = Source_model(img, diversity=False)

        err = (out.data.max(1)[1] != label.data).float().sum()
        natural_err_total += err

        x_adv = PGD20(Source_model, img.clone(), label.clone(), args, loss_fn)
       
        with torch.no_grad():
            out_X_pgd = Source_model(x_adv, diversity=False).detach()
        err_adv = (out_X_pgd.max(1)[1] != label.data).float().sum()

        pgd_err_total += err_adv
        count += batch_szie

        print("Attacked: {}, Batch size: {}, Nature error: {}, Source error: {}".format(count, batch_szie, err.item(), err_adv.item()))


    if args.distributed:
        natural_err_total = reduce_tensor(natural_err_total.data, args.world_size)
        pgd_err_total = reduce_tensor(pgd_err_total.data, args.world_size)

    torch.cuda.synchronize()

    time_end = time.time()

    if not args.distributed or args.local_rank == 0:
        print('time cost', time_end-time_start, 's')
        print("Nature Error total: ", natural_err_total)
        print("Source Success total: ", pgd_err_total)
       
if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = dist.get_world_size()
        opt.batch_size = int(opt.batch_size / opt.world_size)

    if not opt.distributed or opt.local_rank == 0:
        print(opt)
    main(opt)

