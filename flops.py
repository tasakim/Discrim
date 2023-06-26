from thop import profile
import torch
from utils import *

if __name__ == '__main__':

    model = torch.load('./mae/best_c10_resnet56.pt')
    model = model.cuda()
    flops_list = []
    for i in range(64, 0, -1):
        x = torch.rand([1, i, 576]).cuda()
        flops, params = profile(model, inputs=(x, ), verbose=False)
        flops_list.append(flops*1e-6)
    flops_list.reverse()
    print(len(flops_list))

    layer_num = [16] * 9 + [32] * 9 + [64] * 9

    s1_100 = [100 * flops_list[7]] * 9 + [100 * flops_list[15]] * 9 + [100 * flops_list[31]] * 9

    s1_5000 = [5000 * flops_list[7]] * 9 + [5000 * flops_list[15]] * 9 + [5000 * flops_list[31]] * 9

    s2 = [16 * flops_list[14]] * 9 + [32 * flops_list[30]] * 9 + [64 * flops_list[62]] * 9

    s3 = [sum([num * flops_list[num-2] for num in range(16, 8, -1)])] * 9 + \
         [sum([num * flops_list[num-2] for num in range(32, 16, -1)])] * 9  + \
         [sum([num * flops_list[num-2] for num in range(64, 32, -1)])] * 9


    print(s1_100, '\n')
    print(s1_5000, '\n')
    print(s2, '\n')
    print(s3, '\n')