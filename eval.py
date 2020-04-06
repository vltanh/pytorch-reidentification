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
def reid_evaluate(emb_query, emb_gallery, lb_ids_query, lb_ids_gallery,
                  cmc_rank=1, top_k=100):
    # Calculate distance matrix between query images and gallery images
    dist_mtx = pdist_torch(emb_query, emb_gallery).cpu().numpy()
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

    print('Evaluate...')
    cmc_rank = 1
    top_k = 100
    mAP, cmc = reid_evaluate(q_embs, g_embs, q_labels, g_labels,
                             cmc_rank, top_k)
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
