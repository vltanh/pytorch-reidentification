import torch.nn as nn

from .extractor_network import ExtractorNetwork
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor(ExtractorNetwork):
    def __init__(self, version):
        super().__init__()
        assert version in range(9)
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-b{version}')
        self.feature_dim = self.extractor._fc.in_features
        self.return_map = return_map

    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x


class EfficientNetWithAttnExtractor(ExtractorNetwork):
    def __init__(self, version, nheads=8):
        super().__init__()
        assert version in range(9)
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-b{version}')
        self.feature_dim = self.extractor._fc.in_features
        enc = nn.TransformerEncoderLayer(self.feature_dim, nheads)
        self.attn = nn.TransformerEncoder(enc, 1)

    def forward(self, x):
        x = self.extractor.extract_features(x)  # => [B, C, H, W]
        x = x.view(x.size(0), x.size(1), -1)  # => [B, C, HxW]
        x = self.attn(x)  # => [B, C, HxW]
        x = torch.mean(x, dim=1)  # => [B, C]
        return x
