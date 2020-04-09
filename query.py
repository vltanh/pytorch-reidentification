import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from datasets import ImageFolderDataset
from utils.getter import get_instance

import os
import csv
import argparse
from datetime import datetime


@torch.no_grad()
def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


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


def ranking(emb_query, names_query,
            emb_gallery, names_gallery,
            top_k=100):
    # Calculate distance
    dist_mtx = pdist_torch(emb_query, emb_gallery).cpu().numpy()
    # dist_mtx = re_ranking(emb_query, emb_gallery, 60, 6, 0.25)

    # Rank by distance
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis=1)

    reference_path = 'data/AIC20_ReID/test_track.txt'
    reference = open(reference_path).readlines()
    tracks = [x.strip().split() for x in reference if len(x.strip()) != 0]
    tracklet_ids = range(len(tracks))
    veh2tracklet_mapping = {v: i for i, x in enumerate(tracks) for v in x}

    submit = [[] for _ in range(n_q)]
    for qidx in tqdm(range(n_q)):
        # Re-rank
        g_indices = indices[qidx, :]
        match_distances = dist_mtx[qidx, g_indices]
        match_ids = names_gallery[g_indices]

        match_tracklet_ids = np.array([veh2tracklet_mapping[x]
                                       for x in match_ids])
        match_tracklet_distances = np.array([np.median(match_distances[np.where(match_tracklet_ids == tracklet_id)[0]])
                                             for tracklet_id in tracklet_ids])
        tracklet_indices = np.argsort(match_tracklet_distances)

        match_ids = sum([tracks[i] for i in tracklet_indices], [])[:top_k]

        ######################################################################
        qimid = os.path.basename(names_query[qidx]).replace('.jpg', '')
        submit[int(qimid) - 1] = [os.path.basename(x).replace('.jpg', '')
                                  for x in match_ids]

    return submit


def create_submission_file(model_id, save_root, submit):
    save_dir = f'{save_root}/{model_id}/OrgTool'
    submit_path = f'{save_root}/{model_id}/{model_id}-track2.txt'
    os.makedirs(save_dir, exist_ok=True)

    for qimid, result in enumerate(submit):
        save_file = f'{save_dir}/{qimid + 1:06d}.txt'
        with open(save_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([[x] for x in result])

    with open(submit_path, 'w') as g:
        writer = csv.writer(g, delimiter=' ')
        writer.writerows(submit)


@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    image_names = []
    for images, image_name in tqdm(dataloader):
        images = images.to(device)
        embeddings.append(model.get_embedding(images).detach())
        image_names.extend(image_name)
    return torch.cat(embeddings), np.array(image_names)


def generate_submission(gpus, weight_path, save_dir,
                        query_dir, gallery_dir, top_k):
    dev_id = 'cuda:{}'.format(gpus) \
        if torch.cuda.is_available() and gpus is not None \
        else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    assert os.path.exists(weight_path)
    pretrained = torch.load(weight_path, map_location=dev_id)

    model_id = pretrained['config']['id'] + '-' + \
        datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    # 1: Define network
    net = get_instance(pretrained['config']['model']).to(device)
    net.load_state_dict(pretrained['model_state_dict'])

    # 2: Load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print('Load queries...')
    dataset = ImageFolderDataset(query_dir, transform)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract queries...')
    q_embs, q_ids = extract_embeddings(dataloader, net, device)

    print('Load gallery...')
    dataset = ImageFolderDataset(gallery_dir, transform)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract gallery...')
    g_embs, g_ids = extract_embeddings(dataloader, net, device)

    print('Generate...')
    submission = ranking(q_embs, q_ids,
                         g_embs, g_ids, top_k)
    create_submission_file(model_id, save_dir, submission)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight')
    parser.add_argument('--query')
    parser.add_argument('--gallery')
    parser.add_argument('--save', default='vis')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--top', default=100)

    args = parser.parse_args()

    generate_submission(args.gpus, args.weight, args.save,
                        args.query, args.gallery, args.top)
