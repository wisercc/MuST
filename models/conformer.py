import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (22, 1)),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        b, e, h, w = x.size()
        x = x.view(b, e, h * w).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        b, n, _ = x.size()
        q = self.queries(x).view(b, n, self.num_heads, self.emb_size // self.num_heads).transpose(1, 2)
        k = self.keys(x).view(b, n, self.num_heads, self.emb_size // self.num_heads).transpose(1, 2)
        v = self.values(x).view(b, n, self.num_heads, self.emb_size // self.num_heads).transpose(1, 2)

        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        if mask is not None:
            energy = energy.masked_fill(~mask, float('-inf'))

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()
        self.attention_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, num_heads, drop_p),
            nn.Dropout(drop_p)
        ))
        self.feed_forward_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
            nn.Dropout(drop_p)
        ))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.feed_forward_block(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.clshead(x)


class Conformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):

        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return self.classification_head(x)
