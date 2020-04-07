import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from torchnet import meter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from datasets.aic2020track2 import AIC2020Track2
from models.reidentification.triplet_net import TripletNet
from workers.trainer import Trainer
from utils.random_seed import set_seed
from utils.getter import get_instance

import argparse
import time
import os


@torch.no_grad()
def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


@torch.no_grad()
def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
            torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(
                all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


@torch.no_grad()
def reid_evaluate(emb_query, emb_gallery, lb_ids_query, lb_ids_gallery,
                  cmc_rank=1, top_k=100, k1=30, k2=3, l=0.1):
    # Calculate distance matrix between query images and gallery images
    # dist_mtx = pdist_torch(emb_query, emb_gallery).cpu().numpy()
    dist_mtx = re_ranking(emb_query, emb_gallery, k1, k2, l)

    n_q, n_g = dist_mtx.shape
    # sort "gallery index" in "distance" ascending order
    indices = np.argsort(dist_mtx, axis=1)
    matches = lb_ids_gallery[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []

    for qidx in tqdm(range(n_q)):
        match = matches[qidx, :top_k]
        if not np.any(match):
            continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        # basically count all correct prediction < cmc_rank
        all_cmcs.append(cmc[:cmc_rank])

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)

    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype=np.float32)
    cmc = np.mean(all_cmcs, axis=0)

    return mAP, cmc


@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    labels = []
    for images, target in tqdm(dataloader):
        images = images.to(device)
        embeddings.append(model.get_embedding(images))
        labels.append(target.numpy())
    return torch.cat(embeddings), np.concatenate(labels)


def evaluate(gpus, weight_path,
             query_dir, query_label,
             gallery_dir, gallery_label,
             top_k, cmc_rank):
    dev_id = 'cuda:{}'.format(gpus) \
        if torch.cuda.is_available() and gpus is not None \
        else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    assert os.path.exists(weight_path)
    pretrained = torch.load(weight_path, map_location=dev_id)

    # 1: Define network
    net = get_instance(pretrained['config']['model']).to(device)
    net.load_state_dict(pretrained['model_state_dict'])

    # 2: Load datasets
    print('Load queries...')
    dataset = AIC2020Track2(query_dir, query_label, train=True)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract queries...')
    q_embs, q_labels = extract_embeddings(dataloader, net, device)

    print('Load gallery...')
    dataset = AIC2020Track2(root=gallery_dir, path=gallery_label, train=True)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract gallery...')
    g_embs, g_labels = extract_embeddings(dataloader, net, device)

    # import optuna

    # def objective(trial):
    #     k1 = trial.suggest_int('k1', 10, 60)
    #     k2 = trial.suggest_int('k2', 1, 6)
    #     l = trial.suggest_uniform('lambda', 0.1, 0.9)

    #     mAP, _ = reid_evaluate(q_embs, g_embs, q_labels, g_labels,
    #                            cmc_rank, top_k,
    #                            k1, k2, l)
    #     return -mAP

    # study = optuna.create_study()
    # study.optimize(objective, n_trials=100)

    # print(study.best_params)

    print('Evaluate...')
    cmc_rank = 1
    top_k = 100
    mAP, cmc = reid_evaluate(q_embs, g_embs, q_labels, g_labels,
                             cmc_rank, top_k, 55, 6, 0.1)
    print(f'mAP@{top_k}={mAP}, cmc@{cmc_rank}={cmc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight')
    parser.add_argument('--query')
    parser.add_argument('--query_label')
    parser.add_argument('--gallery')
    parser.add_argument('--gallery_label')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--map_top', default=100)
    parser.add_argument('--cmc_top', default=1)

    args = parser.parse_args()

    evaluate(args.gpus, args.weight,
             args.query, args.query_label,
             args.gallery, args.gallery_label,
             args.map_top, args.cmc_top)
