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


def create_submission(model_id, save_dir,
                      emb_query, names_query,
                      emb_gallery, names_gallery,
                      top_k=100):

    save_dir = f'{save_dir}/{model_id}'
    os.makedirs(save_dir, exist_ok=True)

    dist_mtx = pdist_torch(emb_query, emb_gallery).cpu().numpy()
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis=1)

    for qidx in tqdm(range(n_q)):
        qimid = os.path.basename(names_query[qidx]).replace('.jpg', '')
        save_file = save_dir + '/' + qimid + '.txt'

        match_id = names_gallery[indices[qidx, :top_k]]
        with open(save_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([[os.path.basename(x).replace('.jpg', '')]
                              for x in match_id])


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
    dataset = ImageFolderDataset('data/AIC20_ReID/image_query', transform)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract queries...')
    q_embs, q_ids = extract_embeddings(dataloader, net, device)

    print('Load gallery...')
    dataset = ImageFolderDataset('data/AIC20_ReID/image_test', transform)
    dataloader = DataLoader(dataset, batch_size=64)

    print('Extract gallery...')
    g_embs, g_ids = extract_embeddings(dataloader, net, device)

    print('Generate...')
    create_submission(model_id, save_dir,
                      q_embs, q_ids,
                      g_embs, g_ids,
                      top_k)


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
