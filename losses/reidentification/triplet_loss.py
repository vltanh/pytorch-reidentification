import torch.nn as nn
import torch.nn.functional as F

from utils import getter


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, size_average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, output, _):
        anchor, positive, negative = output
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if self.size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = getter.get_instance(selector)

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]]
                        - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]]
                        - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()  # , len(triplets)


class OnlineTripletWithClsLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = getter.get_instance(selector)
        self.cls = nn.Linear(1280, 333)

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]]
                        - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]]
                        - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        triplet_loss = F.relu(ap_distances - an_distances + self.margin)

        classification = self.cls(embeddings)
        cls_loss = F.cross_entropy(classification, target)

        losses = triplet_loss + cls_loss
        return losses.mean()  # , len(triplets)
