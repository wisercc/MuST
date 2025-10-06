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
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Conv2D Layer
        self.depthwiseconv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
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
        self.separableconv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
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


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        padding = (self.kernel_size - 1) * self.dilation

        x_padded = F.pad(x, (padding, 0))
        x = self.conv1(x_padded)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x_padded = F.pad(x, (padding, 0))
        x = self.conv2(x_padded)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        shortcut = self.shortcut(residual)
        if x.size(2) != shortcut.size(2):
            min_len = min(x.size(2), shortcut.size(2))
            x = x[:, :, -min_len:]
            shortcut = shortcut[:, :, -min_len:]

        x = x + shortcut

        return x


class EEGTCNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10,
                 filt=10, dropout=0.2, activation='relu', F1=8, D=2, kernLength=64,
                 dropout_eeg=0.1, dropoutType='Dropout'):
        super(EEGTCNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.nb_classes = nb_classes

        F2 = F1 * D
        self.temporal_dim = Samples // 16

        # EEGNet
        self.eegnet_conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.eegnet_bn1 = nn.BatchNorm2d(F1)

        self.eegnet_depthwiseconv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.eegnet_bn2 = nn.BatchNorm2d(F1 * D)
        self.eegnet_elu = nn.ELU()
        self.eegnet_avgpool1 = nn.AvgPool2d((1, 4))

        # Dropout Type
        if dropoutType == 'SpatialDropout2D':
            self.eegnet_dropout1 = nn.Dropout2d(dropout_eeg)
            self.eegnet_dropout2 = nn.Dropout2d(dropout_eeg)
        else:
            self.eegnet_dropout1 = nn.Dropout(dropout_eeg)
            self.eegnet_dropout2 = nn.Dropout(dropout_eeg)

        self.eegnet_separableconv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.eegnet_bn3 = nn.BatchNorm2d(F2)
        self.eegnet_avgpool2 = nn.AvgPool2d((1, 4))

        self.tcn_layers = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** i
            if i == 0:
                tcn_block = TCNBlock(F2, filt, kernel_s, dilation, dropout)
            else:
                tcn_block = TCNBlock(filt, filt, kernel_s, dilation, dropout)
            self.tcn_layers.append(tcn_block)

        self.classifier = nn.Linear(filt, nb_classes)
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.eegnet_conv1(x)
        x = self.eegnet_bn1(x)

        x = self.eegnet_depthwiseconv(x)
        x = self.eegnet_bn2(x)
        x = self.eegnet_elu(x)
        x = self.eegnet_avgpool1(x)
        x = self.eegnet_dropout1(x)

        x = self.eegnet_separableconv(x)
        x = self.eegnet_bn3(x)
        x = self.eegnet_elu(x)
        x = self.eegnet_avgpool2(x)
        x = self.eegnet_dropout2(x)

        x = x.squeeze(2)

        if x.size(2) != self.temporal_dim:
            x = F.interpolate(x, size=self.temporal_dim, mode='linear', align_corners=False)
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)RE

        x = x[:, :, -1]

        x = self.classifier(x)
        x = self.softmax(x)

        return x
