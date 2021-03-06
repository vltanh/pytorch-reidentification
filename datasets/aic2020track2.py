import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image

import csv
import json
import random


class AIC2020Track2Hard(data.Dataset):
    def __init__(self, root, csv_path, json_path, train, is_cls=False):
        self.train = train
        camera_ids, vehicle_ids = zip(*list(csv.reader(open(csv_path)))[1:])
        reference = json.load(open(json_path))

        labels = list(map(int, vehicle_ids))
        labels_set = set(labels)
        labels_mapping = {k: i for i, k in enumerate(labels_set)}

        if self.train:
            self.tracks = [[root + '/' + x
                            for x in reference[vehicle_id][camera_id]]
                           for camera_id, vehicle_id in zip(camera_ids, vehicle_ids)]
            labels = torch.tensor([labels_mapping[x] if is_cls else x
                                   for x in labels])
        else:
            self.tracks = [root + '/' + x
                           for camera_id, vehicle_id in zip(camera_ids, vehicle_ids)
                           for x in reference[vehicle_id][camera_id]]
            labels = torch.tensor([labels_mapping[int(vehicle_id)] if is_cls else int(vehicle_id)
                                   for camera_id, vehicle_id in zip(camera_ids, vehicle_ids)
                                   for _ in range(len(reference[vehicle_id][camera_id]))])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.labels = labels
        self.data = self.tracks

    def __getitem__(self, index):
        if self.train:
            image_path = random.choice(self.tracks[index])
        else:
            image_path = self.tracks[index]
        im = Image.open(image_path)
        label = self.labels[index]
        return self.transform(im), label

    def __len__(self):
        return len(self.tracks)


class AIC2020Track2(data.Dataset):
    def __init__(self, root, path, train, is_cls=False):
        self.train = train
        image_name, labels = zip(*list(csv.reader(open(path)))[1:])
        self.image_name = [root + '/' + image_path
                           for image_path in image_name]

        labels = list(map(int, labels))
        labels_set = set(labels)
        labels_mapping = {k: i for i, k in enumerate(labels_set)}
        labels = torch.tensor([labels_mapping[x] if is_cls else x
                               for x in labels])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.labels = labels
        self.data = self.image_name

    def __getitem__(self, index):
        image_path = self.image_name[index]
        im = Image.open(image_path)
        label = self.labels[index]
        return self.transform(im), label

    def __len__(self):
        return len(self.image_name)
