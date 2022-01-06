import os
import time
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import torch.distributed as dist

os.environ['TORCH_HOME']='~/.cache/torch/'


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# custom attack
from attacks.linf import DI_fgsm, GA_DI_fgsm, TDI_fgsm, GA_TDI_fgsm, TDMI_fgsm, GA_TDMI_fgsm
from attacks.feature import DI_FSA, DMI_FSA, GA_DMI_FSA, GA_DI_FSA, Feature_Adam_Attack
from perceptual_advex.attacks import ReColorAdvAttack

from utils import MyCustomDataset, get_architecture, Input_diversity, MultiEnsemble
from utils import CrossEntropyLoss, MarginLoss

parser = argparse.ArgumentParser(description='PyTorch Unrestricted Attack')
parser.add_argument('--batch_size', type=int, default=10,metavar='N', help='batch size for attack (default: 30)')
parser.add_argument('--attack_method', type=str, default="GA_TDMI_fgsm")
parser.add_argument('--save_path', type=str, default="tmp/images")
parser.add_argument('--mode', type=str, default="nearest")
parser.add_argument('--loss_fn', type=str, default="ce")
parser.add_argument('--momentum', default=1.0, type=float,help='momentum, (default: 1.0)')
parser.add_argument('--epsilon', default=16, type=float,help='perturbation, (default: 16)')
parser.add_argument('--max_epsilon', default=16, type=float,help='perturbation, (default: 16)')
parser.add_argument('--intervals', default=5, type=int,help='number of intervals')
parser.add_argument('--num_steps', default=10, type=int,help='number of steps in TDMI')
parser.add_argument('--kernel_size', default=5, type=int,help='kernel size of gaussian filter') 
parser.add_argument('--target_id', default=7, type=int,help='InceptionResnetV2 as default')
parser.add_argument('--source_list', default='2_3_5', type=str)
parser.add_argument('--auxiliary_list', default='1_4_6', type=str)
parser.add_argument('--prob', default=0.7, type=float,help='input diversity prob')
parser.add_argument('--thres', default=0.01, type=float, help='threshold for continue attack')
parser.add_argument('--scale', default=0.1, type=float,help='input diversity scale')
parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

NUM_CLASSES = 1000

MODEL_NAME_DICT = {
    0: "vit_small_patch16_224",
    1: "vit_base_patch16_224",
    2: "swin_base_patch4_window7_224",
    3: "swsl_resnext101_32x8d",
    4: "ssl_resnext50_32x4d",
    5: "swsl_resnet50",
    6: "InceptionV3",
    7: "InceptionResnetV2",
    8: "adv_inception_v3",
    9: "ens_adv_inception_resnet_v2",
    10: "Resnet152-DenoiseAll",
    11: "AdvResnet152",
    12: "Resnext101-DenoiseAll"
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

def normalize(item):
    max = item.max()
    min = item.min()
    return (item - min) / (max - min)

def main(args):

    init_seeds(cuda_deterministic=True)

    # Target model
    if not args.distributed or args.local_rank == 0:
        print("Using {} as target model!".format(MODEL_NAME_DICT[args.target_id]))
    tmp_model = get_architecture(model_name=MODEL_NAME_DICT[args.target_id]).cuda().eval()
    Target_model = Input_diversity(tmp_model, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    # Source model
    source_id_list = [int(item) for item in args.source_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        print("Source id list: {}".format(source_id_list))
    
    Source_model_list = []
    for idx in source_id_list:
        temp_model = get_architecture(model_name=MODEL_NAME_DICT[idx]).cuda().eval()
        Source_model_list.append(temp_model)

    Source_model = MultiEnsemble(Source_model_list, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    # Auxiliary model
    auxiliary_id_list = [int(item) for item in args.auxiliary_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        print("Auxiliary id list: {}".format(auxiliary_id_list))

    Auxiliary_model_list = []
    for idx in auxiliary_id_list:
        temp_model = get_architecture(model_name=MODEL_NAME_DICT[idx]).cuda().eval()
        Auxiliary_model_list.append(temp_model)

    Auxiliary_model = MultiEnsemble(Auxiliary_model_list, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    loader = MyCustomDataset(img_path="data/images")
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(loader)
    else:
        sampler = torch.utils.data.SequentialSampler(loader)

    attack_loader = torch.utils.data.DataLoader(dataset=loader,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    diff_path = args.save_path[:-7] + "/diff"
    if not os.path.isdir(diff_path):
        os.makedirs(diff_path, exist_ok=True)

    # save budgrt with a dict
    img_budget = {}

    natural_err_total, pgd_err_total, target_err_total = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    eps_total = torch.tensor(0.0).cuda()
    quality_level = torch.tensor(0.0).cuda()

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

    # distance metric
    if "fgsm" in args.attack_method:
        reward_fn = lambda x: 1.0 / (x * 255.0)
    else:
        reward_fn = lambda x: 1.0 / (x)
        

    for (img, label, img_name) in attack_loader:
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        with torch.no_grad():
            out = Source_model(img, diversity=False)

        err = (out.data.max(1)[1] != label.data).float().sum()
        natural_err_total += err

        if args.attack_method == "TDI_fgsm":
            x_adv, budget = TDI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_TDI_fgsm":
            x_adv, budget = GA_TDI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "DI_fgsm":
            x_adv, budget = DI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DI_fgsm":
            x_adv, budget = GA_DI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "TDMI_fgsm":
            x_adv, budget = TDMI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_TDMI_fgsm":
            x_adv, budget = GA_TDMI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)


        elif args.attack_method == "FSA":
            x_adv = Feature_Adam_Attack(Source_model, img.clone(), label.clone(), args, loss_fn, diversity=False)
            budget = args.epsilon * torch.ones([x_adv.shape[0]])
        elif args.attack_method == "DI_FSA":
            x_adv, budget= DI_FSA(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DI_FSA":
            x_adv, budget = GA_DI_FSA(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "DMI_FSA":
            x_adv, budget = DMI_FSA(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DMI_FSA":
            x_adv, budget = GA_DMI_FSA(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "recolor":
            attack = ReColorAdvAttack(model=Source_model, bound=args.epsilon, num_iterations=args.num_steps)
            x_adv = attack(img.clone(), label.clone()).detach()
            budget = args.epsilon * torch.ones([x_adv.shape[0]])
        else:
            raise Exception("invalid attack method !")
        
        with torch.no_grad():
            out_adv = Source_model(x_adv, diversity=False).detach()
        err_adv = (out_adv.max(1)[1] != label.data).float().sum()

        pgd_err_total += err_adv
        count += batch_szie

        # Attack on Target model
        with torch.no_grad():
            out = Target_model(x_adv, diversity=False)
            err_mask = (out.data.max(1)[1] != label.data)
            err_target = err_mask.float().sum()
            target_err_total += err_target

        distance_batch = budget[err_mask]
        
        if distance_batch.sum() == 0:
            print("Attacked: {}, Batch size: {}, Image name: {}, Nature error: {}, Source error: {}, Target error mask: {}, Batch distance Max: {}, Avg: {}, Avg reward: {}".format(count, batch_szie, img_name, err.item(), err_adv.item(), err_mask, 0, 0, 0.0))
        else:
            eps_total += distance_batch.sum()
            batch_score = reward_fn(distance_batch)
            quality_level += batch_score.sum() 
            print("Attacked: {}, Batch size: {}, Image name: {}, Nature error: {}, Source error: {}, Target error mask: {}, budgets: {}, Batch distance Max: {}, Avg: {}, Avg reward: {}".format(count, batch_szie, img_name, err.item(), err_adv.item(), err_mask, budget, distance_batch.max().item(), distance_batch.mean().item(), batch_score.mean().item()))

        budget_cpu = budget.detach()
        budget_cpu[~err_mask] *= -1
        budget_cpu = budget_cpu.cpu().numpy()

        for i in range(batch_szie):
            x_adv_cpu = x_adv[i, :, :, :].cpu()
            img_adv = transforms.ToPILImage()(x_adv_cpu).convert('RGB')
            img_adv.save(args.save_path + "/" + img_name[i])

            x_cpu_numpy = 255.0 * img[i, :, :, :].cpu().numpy().transpose(1,2,0)
            x_adv_numpy = 255.0 * x_adv_cpu.numpy().transpose(1,2,0)
            save_diff = np.concatenate((x_cpu_numpy, x_adv_numpy, 255.0 * normalize(x_adv_numpy - x_cpu_numpy)))
            save_diff = np.reshape(save_diff, newshape=[299*3, 299, 3])
            save_diff = save_diff.astype(np.uint8)
            Image.fromarray(save_diff, mode='RGB').save(diff_path + "/" + img_name[i])

            # budget
            img_budget[img_name[i]] = budget_cpu[i]

    if args.distributed:
        eps_total = reduce_tensor(eps_total.data, args.world_size)
        quality_level = reduce_tensor(quality_level.data, args.world_size)
        natural_err_total = reduce_tensor(natural_err_total.data, args.world_size)
        pgd_err_total = reduce_tensor(pgd_err_total.data, args.world_size)
        target_err_total = reduce_tensor(target_err_total.data, args.world_size)

        # save budget localrank-wise
        budget_path = os.path.join(args.save_path[:-7], str(args.local_rank) + "_rank_budget.npy")
        np.save(budget_path, img_budget)
    else:
        budget_path = os.path.join(args.save_path[:-7], "budget.npy")
        np.save(budget_path, img_budget)

    torch.cuda.synchronize()
    
    time_end = time.time()
    if not args.distributed or args.local_rank == 0:
        print('time cost', time_end-time_start, 's')
        print("Nature Error total: ", natural_err_total)
        print("Source Success total: ", pgd_err_total)
        print("Target Success total: ", target_err_total)
        if "fgsm" in args.attack_method:
            print('Avg distance of successfully transferred: {}'.format((eps_total / target_err_total) * 255.0))
        else:
            print('Avg distance of successfully transferred: {}'.format((eps_total / target_err_total)))   
        print('Avg perturbation reward: {}'.format((quality_level / target_err_total)))


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

