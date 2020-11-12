import argparse

import scipy
import torch
import numpy as np
import individual_TF
import baselineUtils
from transformer.functional import subsequent_mask


def parser():
    parser = argparse.ArgumentParser(description='Infer using the individual Transformer model')
    parser.add_argument('--dataset_name', type=str, default='nuscenes')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()
    return args


def main(dataset_name, model_layers, emb_size, heads, obs=8, preds=12, dropout=0.):
    device = torch.device("cuda")
    model = individual_TF.IndividualTF(2, 3, 3, N=model_layers,
                                       d_model=emb_size, d_ff=2048, heads=heads, dropout=dropout,
                                       mean=[0, 0], std=[0, 0]).to(device)
    model.load_state_dict(torch.load(f'models/Individual/{dataset_name}/model.pth'))
    mean_std = scipy.io.loadmat(f'models/Individual/{dataset_name}/norm.mat')

    model.eval()

    inference_set, _ = baselineUtils.create_dataset("datasets", args.dataset_name, 0,
                                 obs, preds, verbose=True, inference=True)

    inference_dl = torch.utils.data.DataLoader(inference_set, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    gt = []
    pr = []
    inp_ = []
    peds = []
    frames = []
    dt = []

    for id_b, batch in enumerate(inference_dl):
        inp_.append(batch['src'])
        gt.append(batch['trg'][:, :, 0:2])
        frames.append(batch['frames'])
        peds.append(batch['peds'])
        dt.append(batch['dataset'])

        inp = (batch['src'][:, 1:, 2:4].to(device) - torch.tensor(mean_std["mean"], device=device)) / torch.tensor(mean_std["std"], device=device)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
        dec_inp = start_of_seq

        for i in range(12):
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
            out = model(inp, dec_inp, src_att, trg_att)
            dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

        preds_tr_b = (dec_inp[:, 1:, 0:2] * torch.tensor(mean_std["std"], device=device)
                      + torch.tensor(mean_std["mean"], device=device)).cpu().detach().numpy().cumsum(1) + \
                     batch['src'][:, -1:, 0:2].cpu().numpy()
        pr.append(preds_tr_b)
        print("inference: batch %04i / %04i" % (id_b, len(inference_dl)))

    peds = np.concatenate(peds, 0)
    frames = np.concatenate(frames, 0)
    dt = np.concatenate(dt, 0)
    gt = np.concatenate(gt, 0)
    dt_names = inference_set.data['dataset_name']
    pr = np.concatenate(pr, 0)
    mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
    print(gt[0], pr[0])
    print(mad, fad)

if __name__ == '__main__':
    args = parser()
    main(args.dataset_name, args.layers, args.emb_size, args.heads)