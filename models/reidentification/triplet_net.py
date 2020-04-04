import torch.nn as nn

from utils import getter


class TripletNet(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.embedding_net = getter.get_instance(extractor)

    def forward(self, x):
        x1, x2, x3 = x
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
