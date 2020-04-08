import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from datasets.aic2020track2 import AIC2020Track2


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


@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = torch.zeros(len(dataloader.dataset), model.feature_dim)
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    for images, target in tqdm(dataloader):
        images = images.to(device)
        embeddings[k:k+len(images)
                   ] = model.get_embedding(images).detach()
        labels[k:k+len(images)] = target.numpy()
        k += len(images)
    return embeddings, labels


class MeanAP():
    def __init__(self, net, device,
                 root='data/AIC20_ReID/image_train',
                 path='data/list/reid_query_easy.csv',
                 batch_size=64, top_k=100):
        self.root = root
        self.path = path
        self.top_k = top_k

        self.net = net
        self.device = device
        dataset = AIC2020Track2(root=self.root,
                                path=self.path,
                                train=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size)

        self.reset()

    def calculate(self, output, target):
        self.embeddings.append(output)
        self.labels.append(target)
        return None

    def update(self, value):
        pass

    def reset(self):
        self.embeddings = []
        self.labels = []
        self.score = None

    def value(self):
        g_embs = torch.cat(self.embeddings).to(self.device)
        g_labels = torch.cat(self.labels).cpu().numpy()
        q_embs, q_labels = extract_embeddings(self.dataloader,
                                              self.net, self.device)
        q_embs = q_embs.to(self.device)

        mAP, _ = reid_evaluate(q_embs, g_embs,
                               q_labels, g_labels,
                               top_k=self.top_k)
        self.score = mAP
        return mAP

    def summary(self):
        if self.score is None:
            self.score = self.value()
        print(f'mAP@{self.top_k}={self.score}')
