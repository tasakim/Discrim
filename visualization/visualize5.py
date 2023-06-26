import torch
import torchvision.models
import sys
sys.path.append('..')
from utils import *
from models import *
import matplotlib.pyplot as plt

'''
查看重构后的filter所产生的特征图与原始的差异
'''

# def hook_fn1(module, input, output):
#     for i in range(output.shape[0]):
#         for index in range(output[i].shape[0]):
#             plt.subplot(math.sqrt(output.shape[1]), math.sqrt(output.shape[1]), index + 1)
#             plt.imshow(output[i][index, :, :].detach().cpu())
#             plt.title(index)
#             plt.axis('off')
#         plt.savefig(r'./fig/prune/{}.png'.format(i))
#
#
# def hook_fn2(module, input, output):
#     for i in range(output.shape[0]):
#         for index in range(output[i].shape[0]):
#             plt.subplot(math.sqrt(output.shape[1]), math.sqrt(output.shape[1]), index + 1)
#             plt.imshow(output[i][index, :, :].detach().cpu())
#             plt.axis('off')
#         plt.savefig(r'./fig//target/{}.png'.format(i))

def hook_fn(module, input, output):
    # import pdb
    # pdb.set_trace()
    global layer_index
    global fig_index
    print(fig_index, layer_index)
    reconstructed_weight = torch.zeros_like(module.weight)
    if layer_index in layer_list:
        weight = module.weight
        shape = module.weight.shape
        num_plot = int(shape[0]*p_ratio)
        old_weight = weight.flatten(1)
        length = int(old_weight.shape[1])
        old_weight = torch.nn.ConstantPad1d((0, int(4608 - length)), 0)(old_weight)
        # rand_indices = torch.rand(old_weight.shape[0], device=old_weight.device).argsort(dim=-1)
        loss_list = torch.zeros([shape[0]])
        for index in range(old_weight.shape[0]):
            masked_indices = torch.LongTensor([index])
            unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
            w = old_weight[unmasked_indices, :]
            w = w.unsqueeze(0)
            w = discrim.encoder[0](w)
            w += discrim.pos_emd0(unmasked_indices.cuda())
            w = discrim.encoder[1](w)
            w = discrim.enc_to_dec(w)
            f = torch.zeros([1, old_weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            # f[:, unmasked_indices, :] += discrim.pos_emd(unmasked_indices.cuda())
            # f[:, masked_indices, :] += discrim.pos_emd(masked_indices.cuda())
            f[:, masked_indices, :] += discrim.pos_emd(masked_indices.cuda())
            new_weight = discrim.decoder(f)[:, masked_indices, :]
            loss = F.mse_loss(new_weight[:, :, :length], old_weight[masked_indices, :length].unsqueeze(0))
            loss_list[index] = loss
            # print(new_weight[:, :, :length].shape)
        mask_index = loss_list.topk(k=num_plot, largest=False).indices.squeeze()
        masked_indices = torch.LongTensor(mask_index)
        unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
        w = old_weight[unmasked_indices, :]
        w = w.unsqueeze(0)
        w = discrim.encoder[0](w)
        w += discrim.pos_emd0(unmasked_indices.cuda())
        w = discrim.encoder[1](w)
        w = discrim.enc_to_dec(w)
        f = torch.zeros([1, old_weight.shape[0], w.shape[2]], device=w.device)
        f[:, unmasked_indices, :] = w
        # f[:, unmasked_indices, :] += discrim.pos_emd(unmasked_indices.cuda())
        # f[:, masked_indices, :] += discrim.pos_emd(masked_indices.cuda())
        f[:, masked_indices, :] += discrim.pos_emd(masked_indices.cuda())
        new_weight = discrim.decoder(f)[:, masked_indices, :]
        # import pdb
        # pdb.set_trace()
        reconstructed_weight = new_weight[:, :, :length].reshape([mask_index.shape[0], shape[1], shape[2], shape[3]])


        # old_output = output[:, masked_indices, :, :]
        # new_output = F.conv2d(input[0], new_weight[:, :, :length].reshape(module.weight.shape), bias=module.bias, stride=module.stride, padding=module.padding)[:, masked_indices, :, :]
        old_output = output[:, mask_index, :, :]
        new_output = F.conv2d(input[0], reconstructed_weight, bias=module.bias, stride=module.stride, padding=module.padding)
        # plot_index = ((new_output - old_output)**2).sum(dim=(2,3)).topk(k=num_plot, largest=False).indices.squeeze()

        plt.figure(figsize=(8,2))
        # plt.xticks([]),plt.yticks([])
        for i in range(num_plot):
            plt.subplot(2, num_plot, i + 1)
            plt.imshow(old_output.squeeze(0)[i].detach().cpu())
            plt.axis('off')
        # plt.savefig(r'./fig/{}_old.png'.format(layer_index))
        # plt.clf()
        for i in range(num_plot):
            plt.subplot(2, num_plot, i + num_plot + 1)
            plt.imshow(new_output.squeeze(0)[i].detach().cpu())
            plt.axis('off')
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
        plt.savefig(r'../fig/{}_{}.png'.format(fig_index, layer_index), dpi=300)
        plt.clf()
        plt.close()
    layer_index += 1


# discrim = Discriminator(max_length=576, d_model=576, enc_depth=2, dec_depth=2, n_head=2)
# discrim = Discriminator(max_dim=4608, enc_dim=64, dec_dim=512, enc_depth=8, dec_depth=8, n_head=2)
# discrim.pos_emd = nn.Embedding(2048, 512)
# discrim.load_state_dict(torch.load('../imagenet.pth'))
discrim = torch.load('../mae/best_imagenet_resnet50.pt')
target_model = torchvision.models.resnet50()
target_model.load_state_dict(torch.load('../pretrained/r50_imagenet.pth'))
p_ratio = 0.75
layer_index = 1
fig_index = 0
criterion = nn.MSELoss()
discrim = discrim.cuda() if torch.cuda.is_available() else discrim

batch_size = 1
# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2470, 0.2435, 0.2616]
# test_transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize(mean, std)])
# test_set = datasets.CIFAR10('/ssd/ssd0/n50031076/Dataset/Cifar10', train=False, transform=test_transform, download=True)
# # test_set = datasets.CIFAR10('./Data/Cifar10', train=False, transform=test_transform, download=True)
# num_classes = 10
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
root = '/ssd/ssd0/n50031076/Dataset/ImageNet'
testdir = '/ssd/ssd0/n50031076/Dataset/ImageNet/val'
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
test_set = datasets.ImageFolder(testdir, test_transform)
num_classes = 1000
#
# test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True,
                                          num_workers=4, pin_memory=True)
nprocs = 1

discrim.eval()
# layer_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25, 27, 29, 31, 33, 35, 37,   39, 42, 44, 46, 48, 50, 52, 54, 56] #r56
# layer_list = [2, 4, 6, 9, 11, 14, 16, 19] #r18
# layer_list = [2, 3, 6, 7, 9, 10]
layer_list = [2, 3, 6, 7, 9, 10, 12, 13, 16, 17, 19, 20, 22, 23 , 25, 26, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 48, 49, 51, 52] #r50
# layer_list = [44, 45, 48, 49, 51, 52]
for name, module in target_model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(hook_fn)
with torch.no_grad():
    for index, (input, target) in enumerate(test_loader):
        fig_index = index
        if index >=5:
            break
        target_model = target_model.cuda() if torch.cuda.is_available() else target_model
        input = input.cuda() if torch.cuda.is_available() else input
        out = target_model(input)
        layer_index = 1
