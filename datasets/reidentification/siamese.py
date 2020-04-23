from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from utils import getter


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = getter.get_instance(dataset)

        self.train = self.dataset.train
        self.transform = self.dataset.transform

        self.labels = self.dataset.labels
        self.data = self.dataset
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            positive_pairs = [[i,
                               np.random.choice(
                                   self.label_to_indices[self.labels[i].item()]),
                               1]
                              for i in range(0, len(self.data), 2)]

            negative_pairs = [[i,
                               np.random.choice(self.label_to_indices[
                                   np.random.choice(
                                       list(
                                           self.labels_set - set([self.labels[i].item()]))
                                   )
                               ]),
                               0]
                              for i in range(1, len(self.data), 2)]
            self.pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.data[index]
            label1 = label1.item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(
                        self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(
                    list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(
                    self.label_to_indices[siamese_label])
            img2, _ = self.data[siamese_index]
        else:
            img1, _ = self.data[self.pairs[index][0]]
            img2, _ = self.data[self.pairs[index][1]]
            target = self.pairs[index][2]

        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)
