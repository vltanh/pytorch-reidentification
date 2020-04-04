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
import random
random.seed(3698)


mnist_classes = list(map(str, range(100)))
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(100)]


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()

    return dist_mtx


def reid_evaluate(emb_query, emb_gallery, lb_ids_query, lb_ids_gallery,
                  cmc_rank=1, top_k=100):
    # Calculate distance matrix between query images and gallery images
    dist_mtx = pdist_torch(emb_query, emb_gallery).cpu().detach().numpy()
    n_q, n_g = dist_mtx.shape
    # sort "gallery index" in "distance" ascending order
    indices = np.argsort(dist_mtx, axis=1)
    matches = lb_ids_gallery[indices] == lb_ids_query[:, np.newaxis]
    matches = matches.astype(np.int32)
    all_aps = []
    all_cmcs = []

    for qidx in tqdm(range(n_q)):
        qpid = lb_ids_query[qidx]
        # qcam = lb_cams_query[qidx]

        order = indices[qidx]
        pid_diff = lb_ids_gallery[order] != qpid
        # cam_diff = lb_cams_gallery[order] != qcam
        useful = lb_ids_gallery[order] != -1
        # keep = np.logical_or(pid_diff, cam_diff)
        # keep = np.logical_and(keep, useful)
        # match = matches[qidx][keep]
        match = matches[qidx, :top_k]
        if not np.any(match):
            continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        # basically count all correct prediction < cmc_rannk
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


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    if embeddings.size(1) > 2:
        embeddings = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(100):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0],
                    embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = torch.zeros(len(dataloader.dataset),
                             model.embedding_net.feature_dim)
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    for images, target in tqdm(dataloader):
        images = images.to(device)
        embeddings[k:k+len(images)
                   ] = model.get_embedding(images).detach()
        labels[k:k+len(images)] = target.numpy()
        k += len(images)
    return embeddings, labels


def evaluate(config):
    dev_id = 'cuda:{}'.format(config['gpus']) \
        if torch.cuda.is_available() and config.get('gpus', None) is not None \
        else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config['pretrained']

    assert os.path.exists(pretrained_path)
    pretrained = torch.load(pretrained_path, map_location=dev_id)
    for item in ["model"]:
        config[item] = pretrained["config"][item]

    # 1: Define network
    net = get_instance(config['model']).to(device)
    net.load_state_dict(pretrained['model_state_dict'])

    # 2: Load datasets
    print('Load queries...')
    dataset = AIC2020Track2(root='data/AIC20_ReID/image_train',
                            path='data/list/reid_query_easy.csv',
                            train=True)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract queries...')
    q_embs, q_labels = extract_embeddings(dataloader, net, device)

    print('Load gallery...')
    dataset = AIC2020Track2(root='data/AIC20_ReID/image_train',
                            path='data/list/reid_gallery_easy.csv',
                            train=True)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract gallery...')
    g_embs, g_labels = extract_embeddings(dataloader, net, device)

    print('Evaluate...')
    cmc_rank = 1
    top_k = 100
    mAP, cmc = reid_evaluate(q_embs, g_embs, q_labels, g_labels,
                             cmc_rank, top_k)
    print(f'mAP@{top_k}={mAP}, cmc@{cmc_rank}={cmc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--weight')

    args = parser.parse_args()

    config = dict()
    config['gpus'] = args.gpus
    config['pretrained'] = args.weight

    evaluate(config)
