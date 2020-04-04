import torch.nn as nn

from utils import getter


class SiameseNet(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.embedding_net = getter.get_instance(extractor)

    def forward(self, x):
        x1, x2 = x
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
