import copy
import torch.nn as nn
import torch
from utils import *
from models import *
import matplotlib.pyplot as plt
import math

def hook_fn1(module, input, output):
    for i in range(output.shape[0]):
        for index in range(output[i].shape[0]):
            plt.subplot(math.sqrt(output.shape[1]), math.sqrt(output.shape[1]), index + 1)
            plt.imshow(output[i][index, :, :].detach().cpu())
            plt.title(index)
            plt.axis('off')
        plt.savefig(r'./fig/prune/{}.png'.format(i))


def hook_fn2(module, input, output):
    for i in range(output.shape[0]):
        for index in range(output[i].shape[0]):
            plt.subplot(math.sqrt(output.shape[1]), math.sqrt(output.shape[1]), index + 1)
            plt.imshow(output[i][index, :, :].detach().cpu())
            plt.axis('off')
        plt.savefig(r'./fig//target/{}.png'.format(i))

discrim = Discriminator(max_length=576, d_model=512, enc_depth=6, dec_depth=2, n_head=2)
# discrim.pos_emd = nn.Embedding(16, 512)
discrim.load_state_dict(torch.load('ckpt.pth', map_location='cpu' if not torch.cuda.is_available() else 'cuda'))
target_model = torch.load('../target_model.pt')
pruned_model = copy.deepcopy(target_model)
p_ratio = 0.5
criterion = nn.MSELoss()
weight = pruned_model.layer1[3].conv1.weight.data
shape = weight.shape
length = weight.shape[1] * weight.shape[2] * weight.shape[3]
weight = weight.flatten(1)
weight = nn.ConstantPad1d((0, int(576 - length)), 0)(weight)
weight = weight.cuda() if torch.cuda.is_available() else weight
discrim = discrim.cuda() if torch.cuda.is_available() else discrim
score = []
res_list = []
mask_list = []
num_masked = int(p_ratio * weight.shape[0])

discrim.eval()
with torch.no_grad():
    # s = []
    # for i in range(weight.shape[0]):
    #     masked_indices = torch.LongTensor([i])
    #     unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
    #     w = weight[unmasked_indices, :]
    #     w = w.unsqueeze(0)
    #     # print(w.device, discrim.encoder[1].weight.device)
    #     w = discrim.encoder(w)
    #     f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
    #     f[:, unmasked_indices, :] = w
    #     f[:, masked_indices, :] += discrim.pos_emd(masked_indices).unsqueeze(0)
    #     res = discrim.decoder(f)
    #
    #     imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
    #     # imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
    #     s.append(imp.item())

    for i in range(5000):
        rand_indices = torch.rand(weight.shape[0], device=weight.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:num_masked], rand_indices[num_masked:]
        # print(masked_indices)
        w = weight[unmasked_indices, :]
        w = w.unsqueeze(0)
        # print(w.device, discrim.encoder[1].weight.device)
        w = discrim.encoder(w)
        f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
        f[:, unmasked_indices, :] = w
        # f[:, masked_indices, :] += discrim.pos_emd(masked_indices).unsqueeze(0)
        res = discrim.decoder(f)

        imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
        # imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
        mask_list.append(masked_indices)
        score.append(imp)
        res_list.append(res)
index = score.index(max(score))
w = weight[:, :length] / weight[:, :length].norm(dim=-1, keepdim=True)
print(weight[:, :length].norm(dim=-1).topk(num_masked).indices.sort().values)
print(torch.mm(w, w.transpose(1, 0)).sum(dim=-1).topk(num_masked).indices.sort().values)
print(mask_list[index].sort().values)

res = res_list[index]
pruned_model.layer1[3].conv1.weight.data = res[..., :length].squeeze(0).reshape(shape)
pruned_model.layer1[3].conv1.register_forward_hook(hook_fn1)
target_model.layer1[3].conv1.register_forward_hook(hook_fn2)
batch_size = 8
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])
test_set = datasets.CIFAR10('/ssd/ssd0/n50031076/Dataset/Cifar10', train=False, transform=test_transform, download=True)
# test_set = datasets.CIFAR10('./Data/Cifar10', train=False, transform=test_transform, download=True)
num_classes = 10
nprocs = 1
# test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
# batch_size /= nprocs
test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(batch_size), shuffle=True,
                                          num_workers=0, pin_memory=True
                                          # , sampler=test_sampler
                                          )
# _ = visualize(test_loader, target_model, pruned_model, criterion, args)
#
if torch.cuda.is_available():
    pruned_model.cuda()
    target_model.cuda()
for index, (input, target) in enumerate(test_loader):
    if index >=1:
        break
    input = input.cuda() if torch.cuda.is_available() else input
    out = pruned_model(input)
    out = target_model(input)

