from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from utils import getter


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = getter.get_instance(dataset)

        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(
                                   self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                   np.random.choice(
                                       list(
                                           self.labels_set - set([self.test_labels[i].item()]))
                                   )
                               ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index]
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
            img2, _ = self.train_data[siamese_index]
        else:
            img1, _ = self.test_data[self.test_pairs[index][0]]
            img2, _ = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        return (img1, img2), target

    def __len__(self):
        return len(self.dataset)
