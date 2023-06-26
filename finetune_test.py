import argparse
from utils import *
import time
from models import *
from torch.cuda.amp import GradScaler
from natsort import natsorted
import os

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--data_path', type=str, help='Path to dataset', default='/ssd/ssd0/n50031076/Dataset/Cifar10')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01, help='The Learning Rate.')
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--num_workers', type=int, default=8)

# Checkpoints
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()


def main():
    setup_seed(42)
    recorder = RecorderMeter(args.epochs)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    torch.backends.cudnn.benchmark = True
    ori_model = torch.load('target_model.pt')
    ori_model.cuda(args.local_rank)
    interval = 0
    # print_rank0('--------------Test Original Model--------------')
    # _ = test(test_loader, ori_model, criterion, args)
    # print_rank0('---------------Test Pruned Model---------------')
    # top1, _ = test(test_loader, pruned_model, criterion, args)
    # print_rank0('---------------Pruned Model Acc {}%---------------'.format(top1))
    acc_list = []
    for roots, dir, files in os.walk('./pruned'):
        f = natsorted(files)

    r = 90 // 6 #len(f) // 4
    m = 5
    for i in range(m*r, (m+1)*r):
        best_acc = 0.0
        print_rank0(f[i])
        pruned_model = torch.load('./pruned/{}'.format(f[i]))
        pruned_model.cuda(args.local_rank)
        optimizer, scheduler = prepare_other(pruned_model, args)
        for epoch in range(0, args.epochs):
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            begin = time.time()
            # print_rank0('\n==>>[Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [Time={:.2f}s]'.format(epoch + 1, args.epochs,
            #                                                                                   optimizer.state_dict()[
            #                                                                                       'param_groups'][0][
            #                                                                                       'lr'],
            #                                                                                   interval) \
            #       + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
            #                                                          100 - recorder.max_accuracy(False)))

            train_acc1, train_los1 = train(epoch, train_loader, pruned_model, criterion, optimizer, args)




            # scheduler.step()




            test_top1_2, test_los_2 = test(test_loader, pruned_model, criterion, args)
            is_best = recorder.update(epoch, train_los1, train_acc1, test_los_2, test_top1_2)
            interval = time.time() - begin
            if is_best:
                # if dist.get_rank() == 0:
                #     torch.save(pruned_model.cuda(0), 'best.pt')
                best_acc = test_top1_2
        acc_list.append(best_acc)
        print(acc_list)

if __name__ == '__main__':
    main()
