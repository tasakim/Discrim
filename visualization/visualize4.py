import sys
sys.path.append('..')
from utils import *
from models import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# def draw(name, weight, discrim):
#     '''
#     TSNE降维可视化
#     :param name:
#     :param weight:
#     :param discrim:
#     :return:
#     '''
#     weight = weight.flatten(1)
#     weight_embed = TSNE(n_components=2).fit_transform(weight.cpu().detach().numpy())
#     length = int(weight.shape[1])
#     weight = torch.nn.ConstantPad1d((0, int(discrim.max_dim - length)), 0)(weight)
#     w = weight.unsqueeze(0).cuda()
#     w = discrim.encoder(w)
#     w_embed = TSNE(n_components=2).fit_transform(w.squeeze(0).cpu().detach().numpy())
#     plt.figure()
#     plt.scatter(weight_embed[:, 0], weight_embed[:, 1], c='g')
#     plt.scatter(w_embed[:, 0], w_embed[:, 1], c='r', alpha=0.6)
#     # plt.subplot(1, 2, 1)
#     # plt.scatter(weight_embed[:, 0], weight_embed[:, 1])
#     # plt.title('original weight')
#     # plt.subplot(1, 2, 2)
#     # plt.scatter(w_embed[:, 0], w_embed[:, 1])
#     # plt.title('encoder output')
#     # plt.ylim([-100, 100])
#     # plt.xlim([-100, 100])
#     plt.savefig('./fig/TSNE/{}.png'.format(name))
#     plt.clf()
#     plt.close()
#     print(name)

def draw_matrix(name, weight, discrim):
    '''
    相似度矩阵可视化
    :param name:
    :param weight:
    :param discrim:
    :return:
    '''
    weight = weight.flatten(1)
    matrix1 = torch.mm(weight, weight.t()) / torch.mm(weight.norm(p=2,dim=1,keepdim=True), weight.norm(p=2,dim=1,keepdim=True).t())

    length = int(weight.shape[1])
    weight = torch.nn.ConstantPad1d((0, int(discrim.max_dim - length)), 0)(weight)
    w = weight.unsqueeze(0).cuda()
    w = discrim.encoder(w)
    w = discrim.enc_to_dec(w)
    w += discrim.pos_emd(torch.arange(w.shape[1]).cuda())
    res = discrim.decoder(w)[:, :, :length].squeeze(0)
    matrix2 = torch.mm(res, res.t()) / torch.mm(res.norm(p=2, dim=1, keepdim=True), res.norm(p=2, dim=1, keepdim=True).t())
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(matrix1.cpu().detach().numpy(), vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(matrix2.cpu().detach().numpy(), vmin=0, vmax=1)
    plt.savefig('../fig/Sim_Mat/{}.png'.format(name))
    plt.clf()
    plt.close()
    print(name)

discrim = Discriminator(max_dim=4608, enc_dim=64, dec_dim=512, enc_depth=8, dec_depth=8, n_head=2)
discrim.pos_emd = nn.Embedding(2048, 512)
discrim.load_state_dict(torch.load('../imagenet.pth', map_location='cpu' if not torch.cuda.is_available() else 'cuda'))
discrim = discrim.cuda() if torch.cuda.is_available() else discrim

target_model = torchvision.models.resnet50()
target_model.load_state_dict(torch.load('../pretrained/r50_imagenet.pth'))
if not os.path.exists('../fig/Sim_Mat'):
    os.mkdir('../fig/Sim_Mat')

for name, module in target_model.named_modules():
    if isinstance(module, nn.Conv2d):
        draw_matrix(name, module.weight.data, discrim)


