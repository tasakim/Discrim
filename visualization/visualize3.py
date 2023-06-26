import torch

from utils import *
from models import *
import matplotlib.pyplot as plt
import seaborn as sns
'''
难重构的filter到底提取到了什么特征
'''


def hook_fn(module, input, output):
    # global fig_index
    global layer
    global layer_list
    weight = module.weight
    old_weight = weight.flatten(1)
    length = int(old_weight.shape[1])
    weight = torch.nn.ConstantPad1d((0, int(4608 - length)), 0)(old_weight)
    score_list = []
    for i in range(weight.shape[0]):
        masked_indices = torch.LongTensor([i])
        unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
        w = weight[unmasked_indices, :]
        w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
        w = discrim.encoder(w)
        w = discrim.enc_to_dec(w)
        f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
        f[:, unmasked_indices, :] = w
        f[:, masked_indices, :] += discrim.pos_emd(masked_indices.cuda())
        f[:, unmasked_indices, :] += discrim.pos_emd(unmasked_indices.cuda())
        res = discrim.decoder(f)
        imp = F.mse_loss(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
        score_list.append(imp.item())

    score_order = np.argsort(score_list)[::-1]
    plt.figure()
    plt.xticks([]),plt.yticks([])
    output = torch.clamp(output, min=0)
    for i in range(weight.shape[0]):
        plt.subplot(8, int(weight.shape[0] / 8), i + 1)
        plt.imshow(output.squeeze(0)[score_order[i]].detach().cpu()
                   # , vmin=torch.min(output).item()
                   # , vmax=torch.max(output).item()
                   )
        plt.axis('off')
    plt.savefig('./fig/{}.png'.format(layer_list[layer]), dpi=300)
    # sns.distplot(score_list)
    # if not os.path.exists('./fig/{}'.format('KDE')):
    #     os.mkdir('./fig/{}'.format('KDE'))
    # plt.savefig(r'./fig/{}/{}.png'.format('KDE', layer_list[layer]), dpi=300)
    layer += 1
    plt.clf()
    plt.close()


global discrim
discrim = Discriminator(max_length=4608, enc_dim=64, dec_dim=256, enc_depth=2, dec_depth=4, n_head=1)
discrim.pos_emd = nn.Embedding(2048, 256)
discrim.load_state_dict(torch.load('../ckpt_S_0.5.pth', map_location='cpu' if not torch.cuda.is_available() else 'cuda'))
discrim = discrim.cuda() if torch.cuda.is_available() else discrim

target_model = torch.load('../target_model.pt')
p_ratio = 0.5
layer_index = 1
criterion = nn.MSELoss()

batch_size = 1

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
root = '/ssd/ssd0/n50031076/Dataset/ImageNet'
traindir = os.path.join(root, 'train')
testdir = os.path.join(root, 'val')
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
test_set = datasets.ImageFolder(testdir, test_transform)
num_classes = 1000
#
# test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                          num_workers=4, pin_memory=True)

discrim.eval()
global layer
global layer_list
layer_list = [2, 3, 6, 7, 9, 10,   12, 13, 16, 17, 19, 20, 22, 23,   25, 26, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42,   44, 45,
              48, 49, 51, 52]
# layer = 25
layer = 0
count = 0
fig_index = 1
for name, module in target_model.named_modules():
    if isinstance(module, nn.Conv2d):
        count += 1
        if count in layer_list:
            module.register_forward_hook(hook_fn)
with torch.no_grad():
    for index, (input, target) in enumerate(test_loader):
        if index >= 1:
            break
        target_model = target_model.cuda() if torch.cuda.is_available() else target_model
        input = input.cuda() if torch.cuda.is_available() else input
        out = target_model(input)
