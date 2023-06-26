'''
测试不同epoch的discrim剪枝后的精度，验证是否loss越低剪枝后精度越高
'''

import torch
from prune_utils import *
from utils import *
import argparse
import models
import copy
from thop import profile, clever_format
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/Cifar10')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Choose between Cifar10/100 and ImageNet.')
# parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/ImageNet')
# parser.add_argument('--dataset', type=str, default='imagenet',
#                     help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', type=str, default='resnet56', help='Model Architecture')
parser.add_argument('--pretrained', type=str, default='./pretrained/r56_c10.pth',
                    help='Pretrained Weights of CV Models')
parser.add_argument('--ckpt', type=str, help='Checkpoint of Discriminator')
#--------------------------------------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=256, help='Data Batch size.')
parser.add_argument('--p_ratio', type=float, default=0.5, help='Percent of Total Pruned Filters')
parser.add_argument('--normalize', type=int, default=0, help='Normalization')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--mask', type=int, default=1)
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


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    target_model = models.__dict__[args.arch](num_classes=num_classes)
    target_model.load_state_dict(torch.load(args.pretrained))

    pruned_model = copy.deepcopy(target_model)
    manner_list = {'iter': get_mask_iterative,
                   'replace': get_mask_wreplacement,
                   'woreplace': get_mask_woreplacement}
    criterion_cls = nn.CrossEntropyLoss().cuda(args.local_rank)

    l1, l2, l3, skip, max_length = get_config(args)
    # l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    # l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    ratio_list = [0, 0] + [0.30] * 70
    loss_list = []
    pruned_acc = []
    for root, dirs, files in os.walk('./ckpt'):
        f = natsorted(files)
        # discrim = torch.load('best_c10_resnet56.pt', map_location='cpu')
        # discrim = torch.load('best_imagenet_resnet50.pt', map_location='cpu')
    # indices = np.sort(np.random.choice(list(range(len(f))), 50, replace=False)).tolist()
    indices = [i for i in range(50)]

    # for i in range(len(f)):
    for i in indices:
        print_rank0(str(f[i]))
        ckpt = torch.load('./ckpt/{}'.format(f[i]), map_location='cpu')
        discrim = ckpt['weight']
        ckpt_loss = ckpt['loss']
        loss_list.append(ckpt_loss)
        discrim = discrim.cuda()
        max_dim, max_length = discrim.max_dim, max_length
        criterion_rec = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()
        discrim = discrim.cuda() if torch.cuda.is_available() else discrim
        # print_rank0(''.join(['-' * 20, 'Start Pruning', '-' * 20]))
        pruned_model = copy.deepcopy(target_model)
        layer_score_id = 0
        conv_count = 1
        mask_index = []
        for (name, module) in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if conv_count in l1:
                    # print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                    # mask_index = [conv_count]  # r56
                    if args.mask:
                        ratio = 1 - ratio_list[conv_count]
                    else:
                        ratio = ratio_list[conv_count]
                    unmask, mask = manner_list[args.manner](module.weight.data, discrim, criterion_rec, max_dim, ratio, args)
                    mask_index.append(unmask.cpu() if not args.mask else mask.cpu())
                    # mask_index.append(mask.cpu())
                    # print(mask_index[-1].device, module.weight.data.device)
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
                    # print_rank0('---------------Pruning Layer {}---------------'.format(conv_count))
                    if args.mask:
                        ratio = 1 - ratio_list[conv_count]
                    else:
                        ratio = ratio_list[conv_count]
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
                    # print(new_layer)

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
                    # pruned_model.cuda(args.local_rank)
                    # top1, _ = test(test_loader, pruned_model, criterion_cls, args)
                    # print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))

                    # optimizer_cls, scheduler_cls = prepare_other(pruned_model, args)
                    # for finetune_epoch in range(0):  # !!!!!!
                    #     train_sampler.set_epoch(finetune_epoch)
                    #     # import pudb;
                    #     # pu.db
                    #     _ = train(finetune_epoch, train_loader, pruned_model, criterion_cls, optimizer_cls, args)
                    # top1, _ = test(test_loader, pruned_model, criterion_cls, args)
                    # print_rank0('---------------Finetuned Model Acc {}%---------------'.format(top1))
                    # print('*' * 40)
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

        pruned_model.cuda(args.local_rank)
        top1, _ = test(test_loader, pruned_model, criterion_cls, args)
        print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
        pruned_model.cpu()
        pruned_acc.append(top1)
        torch.save(pruned_model, './pruned/pruned_model_{}.pt'.format(i))
    print(loss_list, '\n', pruned_acc)
    # if dist.get_rank() == 0:
    #     print(pruned_model)
    #     size = 32 if num_classes in [10, 100] else 224
    #     ori_flops, ori_params = clever_format(profile(target_model, inputs=(torch.randn(1, 3, size, size),)), '%.2f')
    #     pruned_flops, pruned_params = clever_format(profile(pruned_model, inputs=(torch.randn(1, 3, size, size),)),
    #                                                 '%.2f')
    #     print(ori_flops, ori_params)
    #     print(pruned_flops, pruned_params)
    #     torch.save(target_model, 'target_model.pt')
    #     torch.save(pruned_model, 'pruned_model_{}.pt'.format(ratio_list[2]))


# 唐良智 3516芯片
