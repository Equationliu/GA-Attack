import os
import time
import random
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

# os.environ['TORCH_HOME']='~/.cache/torch/'
os.environ['TORCH_HOME']='/home/data/equation/submit_template_for_AI_challenger_sea6/ckpt/'

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

from utils import MyCustomDataset, get_architecture, Input_diversity

parser = argparse.ArgumentParser(description='Benchmarking ImageNet models under transfer-based unrestricted attacks')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='batch size for attack (default: 30)')
parser.add_argument('--input_path', type=str, default="data/images", help='The path of adversarial examples to evaluate on the benchmark')
parser.add_argument('--mode', type=str, default="nearest")
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

    # Target model
    target_id_list = list(range(0, 10))
    if not args.distributed or args.local_rank == 0:
        print("Target id list: {}".format(target_id_list))
    
    Target_model_list = []
    for idx in target_id_list:
        temp_model = get_architecture(model_name=MODEL_NAME_DICT[idx]).cuda().eval()
        temp_model = Input_diversity(temp_model, args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)
        Target_model_list.append(temp_model)

    loader = MyCustomDataset(img_path=args.input_path)
    
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(loader)
    else:
        sampler = torch.utils.data.SequentialSampler(loader)

    attack_loader = torch.utils.data.DataLoader(dataset=loader,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)
    
    target_err_total = torch.zeros((10, )).cuda()

    count = 0
    time_start = time.time()
    if not args.distributed or args.local_rank == 0:
        print("Starting time counting: {}\n".format(time_start))


    for (img, label, img_name) in attack_loader:
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        count += batch_szie

        target_err_now = []
        with torch.no_grad():
            for idx in range(0, 10):
                out_target = Target_model_list[idx](img, diversity=False).view(batch_szie, -1)
                mask = (out_target.data.max(1)[1] != label.data)
                err_target = mask.float().sum()
                target_err_now.append(err_target.item())
                target_err_total[idx] += err_target

        print("img_id: {}, Attacked: {}, Batch size: {}, Target error: {}".format(img_name, count, batch_szie, target_err_now))

    if args.distributed:
        target_err_total = reduce_tensor(target_err_total.data, args.world_size)

    torch.cuda.synchronize()

    time_end = time.time()

    if not args.distributed or args.local_rank == 0:
        print('time cost', time_end-time_start, 's')
        print("Target Success total: ", target_err_total)
        print("Benchmarking Accuracy: ", [(1000 - item.item()) / 1000 for item in target_err_total])


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

