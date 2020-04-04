import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image

import csv


class AIC2020Track2(data.Dataset):
    def __init__(self, root, path, train):
        self.train = train
        image_name, labels = zip(*list(csv.reader(open(path)))[1:])
        self.image_name = [root + '/' + image_path
                           for image_path in image_name]
        labels = torch.tensor(list(map(int, labels)))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if self.train:
            self.train_labels = labels
            self.train_data = self.image_name
        else:
            self.test_labels = labels
            self.test_data = self.image_name

    def __getitem__(self, index):
        image_path = self.image_name[index]
        im = Image.open(image_path)
        if self.train:
            label = self.train_labels[index]
        else:
            label = self.test_labels[index]
        return self.transform(im), label

    def __len__(self):
        return len(self.image_name)
