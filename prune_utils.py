import torch
import torch.nn as nn
from utils import WData, print_rank0
import numpy as np

def get_mask_iterative(weight, model, criterion, max_length, p_ratio, args):
    score_list = []
    unmask_list = []
    mask_list = []
    model = model.eval()
    num_iter = args.num_iter  # !!!!!!!!!!!!!
    with torch.no_grad():
        length = weight.shape[1] * weight.shape[2] * weight.shape[3]
        weight = weight.flatten(1)
        weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
        weight = weight.cuda() if torch.cuda.is_available() else weight
        num_masked = int(p_ratio * weight.shape[0])
        for iter in range(num_iter):
            rand_indices = torch.rand(weight.shape[0], device=weight.device).argsort(dim=-1)
            masked_indices, unmasked_indices = rand_indices[:num_masked], rand_indices[num_masked:]
            w = weight[unmasked_indices, :]
            w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
            w = model.encoder(w)
            w = model.enc_to_dec(w)

            f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
            f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
            res = model.decoder(f)
            # imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            # print(imp.item())
            score_list.append(imp.item())
            unmask_list.append(unmasked_indices)
            mask_list.append(masked_indices)
        # print(max(score_list), min(score_list))
        unmask = unmask_list[score_list.index(min(score_list))]
        mask = mask_list[score_list.index(max(score_list))]
    return torch.LongTensor(unmask.cpu()), torch.LongTensor(mask.cpu())
    # return torch.LongTensor(mask)


def get_mask_wreplacement(weight, model, criterion, max_length, p_ratio, args):
    score_list = []
    model = model.eval()
    length = weight.shape[1] * weight.shape[2] * weight.shape[3]
    weight = weight.flatten(1)
    weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
    weight = weight.cuda() if torch.cuda.is_available() else weight
    num_masked = int(p_ratio * weight.shape[0])
    with torch.no_grad():
        for i in range(weight.shape[0]):
            masked_indices = torch.LongTensor([i])
            unmasked_indices = torch.LongTensor(list(set(range(weight.shape[0])) - set(masked_indices.tolist())))
            w = weight[unmasked_indices, :]
            w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
            w = model.encoder[0](w)
            w += model.pos_emd0(unmasked_indices.cuda())
            w = model.encoder[1](w)
            w = model.enc_to_dec(w)
            f = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
            f[:, unmasked_indices, :] = w
            f[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
            # f[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
            res = model.decoder(f)
            imp = criterion(res[:, masked_indices, :length], weight.unsqueeze(0)[:, masked_indices, :length])
            # imp = criterion(res[:, :, :length], weight.unsqueeze(0)[:, :, :length])
            # print(imp.item())
            score_list.append(imp.item())
        # print_rank0(', '.join([str(max(score_list)), str(min(score_list))]))

    mask = np.argsort(score_list)[weight.shape[0] - num_masked:]
    # threshold = np.mean(score_list)
    # mask = torch.LongTensor(np.nonzero(np.array(score_list) > threshold)[0])
    unmask = np.argsort(score_list)[:num_masked]
    return torch.LongTensor(unmask), torch.LongTensor(mask)

def get_mask_woreplacement(weight, model, criterion, max_length, p_ratio, args):
    score_list = []
    unmask_list = list(range(weight.shape[0]))
    mask_list = []
    model = model.eval()
    length = weight.shape[1] * weight.shape[2] * weight.shape[3]
    weight = weight.flatten(1)
    weight = nn.ConstantPad1d((0, int(max_length - length)), 0)(weight)
    weight = weight.cuda() if torch.cuda.is_available() else weight
    num_masked = int(p_ratio * weight.shape[0])
    n = 1 #max(int(num_masked * 0.1), 1)
    # print(n)
    shang, yushu = num_masked // n, num_masked % n
    if yushu == 0:
        round = shang
    else:
        round = shang + 1
    with torch.no_grad():
        for i in range(round):
            di = list(set(unmask_list))
            try:
                di.remove(-1)
            except:
                pass
            for j in range(len(di)):
                masked_indices = torch.LongTensor([di[j]])
                unmasked_indices = torch.LongTensor(list(set(di) - set(masked_indices.tolist())))
                # print(unmasked_indices)
                w = weight[unmasked_indices, :]
                tgt = weight[masked_indices, :]
                w = w.unsqueeze(0).cuda() if torch.cuda.is_available() else w.unsqueeze(0)
                w = model.encoder(w)
                w = model.enc_to_dec(w)
                fill = torch.zeros([1, weight.shape[0], w.shape[2]], device=w.device)
                fill[:, unmasked_indices, :] = w
                fill[:, masked_indices, :] += model.pos_emd(masked_indices.cuda())
                fill[:, unmasked_indices, :] += model.pos_emd(unmasked_indices.cuda())
                # f = fill[:, torch.cat((masked_indices, unmasked_indices), dim=-1), :]
                f = fill[:, torch.LongTensor(di), :]
                res = model.decoder(f)
                imp = criterion(res[:, torch.LongTensor([j]), :length], tgt.unsqueeze(0)[:, :, :length])
                # print(imp.item())
                score_list.append(imp.item())
            # mask_list.extend(di[torch.topk(torch.Tensor(score_list), int(s *num_masked)).indices.tolist()])
            if i != round:
                mask_list.extend(np.array(di)[torch.topk(torch.Tensor(score_list), n).indices.tolist()].tolist())
                unmask_list = np.array(unmask_list)
                unmask_list[mask_list[-n]] = -1
            else:
                mask_list.extend(np.array(di)[torch.topk(torch.Tensor(score_list), yushu).indices.tolist()].tolist())
                unmask_list = np.array(unmask_list)
                unmask_list[mask_list[-yushu]] = -1
            unmask_list = unmask_list.tolist()
            score_list = []
    unmask = []
    # print(sorted(mask_list))
    return torch.LongTensor(unmask), torch.LongTensor(mask_list)

