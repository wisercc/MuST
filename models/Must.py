import torch
import torch.nn as nn
from torch import Tensor

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),
            nn.Conv2d(40, 40, (22, 1)),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.AvgPool2d((1, 25), (1, 25)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        b, e, h, w = x.size()
        x = x.view(b, e, h * w).transpose(1, 2)

        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(2*dim)

    def forward(self, x):
        B, L, C = x.shape

        assert L % 2==0, 'The number of patch can not been divided by 2.'

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C

        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(self.reduction(x))

        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return x[..., :x.size(-1) - self.padding[0]]

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(1, expansion, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(emb_size)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        self.conv2 = CausalConv1d(expansion, 1, kernel_size=3, dilation=dilation)

    def forward(self, x):
        b, n, e = x.size()
        x = x.reshape(b * n, 1, e)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.reshape(b, n, e)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, dilation, num_heads=2, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=drop_p, batch_first=True)
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(emb_size)
        self.feed_forward_block = nn.Sequential(
            FeedForwardBlock(emb_size, forward_expansion, forward_drop_p, dilation),
            nn.Dropout(drop_p),
            nn.LayerNorm(emb_size)
        )

    def forward(self, x):
        short_cut = x
        x, _ = self.attention(x,x,x)
        x = self.norm(self.drop(x)) + short_cut
        x = self.feed_forward_block(x) + x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads=2):
        super().__init__()
        self.stage_num = len(depth)
        self.layers = nn.ModuleList()
        for i in range(0,self.stage_num):
            for layer in range(0,depth[i]):
                self.layers.append(TransformerEncoderBlock(emb_size*(2**i), num_heads=num_heads*(2**i), dilation=2**i))
            self.layers.append(PatchMerging(emb_size*(2**i)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes, drop_p=0.5):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(drop_p),
            nn.Linear(emb_size, n_classes)
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.clshead(x)

class Must(nn.Module):
    def __init__(self, emb_size=40, depth=[2,2,2], num_heads=1, n_classes=4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size=emb_size)
        self.transformer_encoder = TransformerEncoder(depth, emb_size, num_heads=num_heads)
        self.classification_head = ClassificationHead(emb_size*2**(len(depth)), n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return self.classification_head(x)
