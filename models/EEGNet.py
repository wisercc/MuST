import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=None, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if F2 is None:
            F2 = F1 * D

        # First Conv2D Layer
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Conv2D Layer
        self.depthwiseconv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False, padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))

        # Dropout Type
        if dropoutType == 'SpatialDropout2D':
            self.dropout1 = nn.Dropout2d(dropoutRate)
            self.dropout2 = nn.Dropout2d(dropoutRate)
        elif dropoutType == 'Dropout':
            self.dropout1 = nn.Dropout(dropoutRate)
            self.dropout2 = nn.Dropout(dropoutRate)
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        # Separable Conv2D Layer
        self.separableconv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 4))

        # Dense Layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * ((Samples // 16) * 1), nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseconv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Second block
        x = self.separableconv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # Fully connected layer
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)

        return x

