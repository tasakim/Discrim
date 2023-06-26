'''
随机剪枝baseline
'''

import argparse
import copy

from utils import *
import time
import models
import random
# from models.cifar.resnet import CifarResNet

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/ImageNet')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='resnet56', help='Model Architecture')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--pretrained', type=str, default='./pretrained/r56_c10.pth', help='Pretrained Weights')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--mode', type=str, default='layerwise', help='Pruning Mode')
# Checkpoints
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()
# args.master_port = random.randint(30000, 40000)

def prune(pruned_model, l, args):
    l1, l2, l3, skip = l['l1'], l['l2'], l['l3'], l['skip']
    layer_score_id = 0
    conv_count = 1
    mask_index = []
    ratio = args.p_ratio
    for (name, module) in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            if conv_count in l1:
                # mask_index.append(cal_score(module.weight.data, threshold, model, criterion, max_length))
                random_mask = np.random.choice(module.weight.shape[0], int((1-ratio)*module.weight.shape[0]), replace=False).astype(np.int64)
                mask_index.append(random_mask)
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                bias = torch.index_select(module.bias.data, dim=0,
                                          index=torch.LongTensor(mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                conv_count += 1
            elif conv_count in l2:
                # mask_index.append(cal_score(module.weight.data, threshold, model, criterion, max_length))
                random_mask = np.random.choice(module.weight.shape[0], int(ratio * module.weight.shape[0]),
                                               replace=False).astype(np.int64)
                mask_index.append(torch.Tensor(random_mask))
                # mask_index.append(torch.Tensor(all_score[layer_score_id] >= threshold).nonzero().squeeze(1))
                # mask_index.append(torch.Tensor(layer_score[layer_score_id] > np.array(sorted(layer_score[layer_score_id])[:int(len(layer_score[layer_score_id]) * ratio)][-1])).nonzero().squeeze())
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                weight = torch.index_select(weight, dim=1, index=torch.LongTensor(mask_index[-1 - 1]))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(
                    mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l3:
                weight = torch.index_select(module.weight.data, dim=1, index=torch.LongTensor(mask_index[-1]))
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                conv_count += 1
                layer_score_id += 1
            elif conv_count in skip:
                conv_count += 1
                layer_score_id += 1
            else:
                conv_count += 1
        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                running_mean = torch.index_select(module.running_mean.data, dim=0,
                                                  index=torch.LongTensor(mask_index[-1]))
                running_var = torch.index_select(module.running_var.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                new_layer = nn.BatchNorm2d(num_features=weight.shape[0])
                new_layer.weight.data = nn.Parameter(weight)
                new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                new_layer.running_mean = running_mean
                new_layer.running_var = running_var
                set_module(pruned_model, name, new_layer)
    return pruned_model



def main():
    setup_seed(42)
    target_model = models.__dict__[args.arch](num_classes=10)
    target_model.load_state_dict(torch.load(args.pretrained, map_location='cpu' if not torch.cuda.is_available() else None))
    #
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    _, test_loader, num_classes, _, test_sampler = prepare_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    torch.backends.cudnn.benchmark = True

    if args.arch == 'resnet56':
        l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 42, 44, 46, 48, 50, 52, 54, 56]
        l2 = []
        l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 32, 34, 36, 38, 40, 43, 45, 47, 49, 51, 53, 55, 57]
        skip = [22, 41]
        l = {'l1': l1, 'l2': l2, 'l3': l3, 'skip': skip}

    for i in range(10):
        model = copy.deepcopy(target_model)
        pruned_model = prune(model, l, args)
        # print(pruned_model)
        pruned_model.cuda(args.local_rank)
        # print_rank0('---------------Test Pruned Model---------------')
        top1, _ = test(test_loader, pruned_model, criterion, args)
        print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
        pruned_model.cpu()

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 finetune.py \
#         --checkpoint pruned_model.pt --dataset cifar10 --data_path /ssd/ssd0/n50031076/Dataset/Cifar10 \
#         --epochs 400 --batch_size 64 --lr 0.01 --num_workers 8