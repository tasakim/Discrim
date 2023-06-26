import numpy as np

from prune_utils import get_mask_wreplacement, get_mask_iterative, get_mask_woreplacement
from utils import *
import argparse
import models
import copy
from thop import profile, clever_format

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/ssd/ssd0/n50031076/Dataset/Cifar10')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='resnet56', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./pretrained/r56_c10.pth')
# parser.add_argument('--data_path', type=str, default='/ssd/ssd0/n50031076/Dataset/ImageNet')
# parser.add_argument('--dataset', type=str, default='imagenet')
# parser.add_argument('--pretrained', type=str, default='./pretrained/r50_imagenet.pth')
parser.add_argument('--ckpt', type=str, help='Checkpoint of Discriminator')
#--------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Data Batch size.')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--manner', type=str, default='replace')
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

print(args)
seed = 42
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if args.arch == 'resnet56':
    base_acc = 94.03
elif args.arch == 'resnet50':
    base_acc = 76.15


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    target_model = models.__dict__[args.arch](num_classes=num_classes)
    target_model.load_state_dict(torch.load(args.pretrained))

    ll1 = list(zip(l1, (np.array(l1) + 1).tolist()))
    ll2 = list(zip(l2, (np.array(l2) + 1).tolist()))
    combi = [data for item in zip(ll1 , ll2) for data in item]
    print(combi)
    manner_list = {'replace': get_mask_wreplacement,
                   'woreplace': get_mask_woreplacement,
                    'iter': get_mask_iterative}
    criterion_cls = nn.CrossEntropyLoss().cuda(args.local_rank)


    ratio_list = [0, 0] + [0.6] * 60
    discrim = torch.load('best_c10_resnet56.pt', map_location='cpu')
    discrim = discrim.cuda()
    max_dim = discrim.max_dim
    criterion_rec = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    discrim = discrim.cuda() if torch.cuda.is_available() else discrim
    print_rank0(''.join(['-' * 20, 'Start Pruning', '-' * 20]))
    acc_drop_list = []
    for round in range(len(combi)):
        print_rank0('Round {}'.format(round))
        l1 = [combi[round][0]]
        l2 = []
        l3 = [combi[round][1]]
        skip = []
        pruned_model = copy.deepcopy(target_model)
        layer_score_id = 0
        conv_count = 1
        mask_index = []

        for (name, module) in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if conv_count in l1:
                    print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                    ratio = 1 - ratio_list[conv_count]
                    unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio, args)
                    mask_index.append(unmask.cpu() if not args.mask else mask.cpu())
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
                    # print(new_layer)
                elif conv_count in l2:
                    print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                    ratio = 1 - ratio_list[conv_count]
                    unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio, args)
                    mask_index.append(unmask.cpu() if not args.mask else mask.cpu())
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
                    pruned_model.cuda(args.local_rank)
                    top1, _ = test(test_loader, pruned_model, criterion_cls, args)
                    print_rank0('---------------Acc Drop {}%---------------'.format(base_acc - top1))
                    acc_drop_list.append(np.round(base_acc - top1, 2))
                    pruned_model.cpu()
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

    print(acc_drop_list)
