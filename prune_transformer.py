import torch
from prune_utils import *
from utils import *
import argparse
import models
import copy
from thop import profile, clever_format
import transformers


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/ImageNet')
parser.add_argument('--dataset', type=str, default='imagenet', help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='deit', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./pretrained/deit_base',
                    help='Pretrained Weights of CV Models')
parser.add_argument('--ckpt', type=str, help='Checkpoint of Discriminator')
#--------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Data Batch size.')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--manner', type=str, default='replace')
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--num_iter', type=int, default=100)

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


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    if 'deit' in args.arch:
        target_model = transformers.DeiTForImageClassification.from_pretrained(args.pretrained)
    else:
        pass

    pruned_model = copy.deepcopy(target_model)
    manner_list = {'iter': get_mask_iterative,
                   'replace': get_mask_wreplacement,
                   'woreplace': get_mask_woreplacement}
    criterion_cls = nn.CrossEntropyLoss().cuda(args.local_rank)
    layer_score_id = 0
    linear_count = 1
    mask_index = []
    l1, l2, l3, skip, max_length = get_config(args)

    ratio_list = [0, 0] + [0.5] * 70
    discrim = torch.load(args.ckpt, map_location='cpu')
    discrim = discrim.cuda()
    max_dim, max_length = discrim.max_dim, max_length
    criterion_rec = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
    discrim = discrim.cuda() if torch.cuda.is_available() else discrim
    print_rank0(''.join(['-' * 20, 'Start Pruning', '-' * 20]))

    for (name, module) in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            if linear_count in l1:
                print_rank0('---------------Pruning Layer {}---------------'.format(linear_count))
                # mask_index = [linear_count]  # r56
                ratio = 1 - ratio_list[linear_count]
                unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio, args)
                mask_index.append(mask.cpu())
                # mask_index.append(mask.cpu())
                # print(mask_index[-1].device, module.weight.data.device)
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                bias = torch.index_select(module.bias.data, dim=0,
                                          index=torch.LongTensor(mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Linear(in_features=weight.shape[1], out_features=weight.shape[0], bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                linear_count += 1

            elif linear_count in l2:
                print_rank0('---------------Pruning Layer {}---------------'.format(linear_count))
                ratio = 1 - ratio_list[linear_count]
                unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio, args)
                mask_index.append(mask.cpu())
                # mask_index.append(torch.Tensor(layer_score[layer_score_id] > np.array(sorted(layer_score[layer_score_id])[:int(len(layer_score[layer_score_id]) * ratio)][-1])).nonzero().squeeze())
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(mask_index[-1]))
                weight = torch.index_select(weight, dim=1, index=torch.LongTensor(mask_index[-1 - 1]))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(
                    mask_index[-1])) if module.bias is not None else None
                new_layer = nn.Linear(in_features=weight.shape[1], out_features=weight.shape[0], bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                layer_score_id += 1
                linear_count += 1
                # print(new_layer)

            elif linear_count in l3:
                weight = torch.index_select(module.weight.data, dim=1, index=torch.LongTensor(mask_index[-1]))
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Linear(in_features=weight.shape[1], out_features=weight.shape[0], bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(pruned_model, name, new_layer)
                linear_count += 1
                layer_score_id += 1
                if args.test:
                    pruned_model.cuda(args.local_rank)
                    top1, _ = test(test_loader, pruned_model, criterion_cls, args)
                    print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
                pruned_model.cpu()
            elif linear_count in skip:
                linear_count += 1
                layer_score_id += 1
            else:
                linear_count += 1

    # pruned_model.cuda(args.local_rank)
    # top1, _ = test(test_loader, pruned_model, criterion_cls, args)
    # print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
    # pruned_model.cpu()

    if dist.get_rank() == 0:
        print(pruned_model)
        size = 32 if num_classes in [10, 100] else 224
        ori_flops, ori_params = clever_format(profile(target_model, inputs=(torch.randn(1, 3, size, size),)), '%.2f')
        pruned_flops, pruned_params = clever_format(profile(pruned_model, inputs=(torch.randn(1, 3, size, size),)),
                                                    '%.2f')
        print(ori_flops, ori_params)
        print(pruned_flops, pruned_params)
        torch.save(target_model, 'target_model.pt')
        torch.save(pruned_model, 'pruned_model_{}.pt'.format(ratio_list[2]))


# 唐良智 3516芯片
