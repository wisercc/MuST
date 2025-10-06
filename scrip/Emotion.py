import os
# os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('CUDA_VISIBLE_DEVICES', '0')

import torch
from torch.utils.data import DataLoader,Subset,ConcatDataset, DataLoader, random_split

from torch.optim import Adam
import torch.nn as nn
import logging
from utils import SeedDataset
import random
import numpy as np
from models.EEGNet import EEGNet
# from models.conformer import Conformer
# from models.Must_woPW import Must
from models.Must import Must
# from models.EEGViT import EEGViT
# from models.EEGDeformer import Deformer
from models.EEGSwin import SwinTransformer1D

import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--sub', type=int, required=True, help='Subject identifier')
args = parser.parse_args()

sub = args.sub

Net = 'Must'

logging.basicConfig(filename='results/Emotion/'+Net+'/training_log_' + str(sub) + '.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数
batch_size = 128
learning_rate = 2e-4
weight_decay = 0
num_epochs = 200
warmup_epochs = 0  # Warm-up阶段的epoch数
n_classes = 4

seed_n = 1234
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

train_dataset = SeedDataset(sub,'T')
val_dataset = SeedDataset(sub,'E')

combined_dataset = ConcatDataset([train_dataset, val_dataset])

# 计算长度
total_size = len(combined_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# 按照8:2的比例划分数据集
train_subset, val_subset = random_split(combined_dataset, [train_size, val_size])

# 你可以现在将这些子集用于数据加载器
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# model = SwinTransformer1D(num_classes=4,in_chans=62,embed_dim=40,depths=[2,2,2],num_heads=[2,4,8],window_size=5,mlp_ratio=4).to(device)
model = Must(n_classes=3,emb_size=70).to(device)
# model = EEGNet(nb_classes=3,Chans=62,Samples=200).to(device)
# model = Conformer().to(device)
# model = EEGViT(num_chan=62, num_time=200, num_patches=8, num_classes=3).to(device)
# model = Deformer(num_chan=62, num_time=200, temporal_kernel=11, num_kernel=40,
#                  num_classes=4, depth=6, heads=10,
#                  mlp_dim=16, dim_head=16, dropout=0.5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), betas=(0.5, 0.99), lr=learning_rate, weight_decay=weight_decay)


# 学习率调度器
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.last_epoch / self.warmup_epochs)
            return [lr for _ in self.base_lrs]
        else:
            return self.base_lrs


warmup_scheduler = WarmUpLR(optimizer, warmup_epochs, 1e-4, learning_rate)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-4)


def train(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    lens = 0.0
    for data, labels in loader:
        data = data.unsqueeze(1)

        lens += len(labels)
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / lens
    scheduler.step()
    return avg_loss, accuracy


from sklearn.metrics import confusion_matrix


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_predictions = []
    conf_matrix = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.unsqueeze(1)
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            # 收集所有标签和预测，用于后续计算混淆矩阵
            # all_labels.extend(labels.cpu().numpy())
            # all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    # conf_matrix = confusion_matrix(all_labels, all_predictions)

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy, conf_matrix

# 在训练循环开始前，初始化一个最佳验证准确率变量
best_val_accuracy = 0.0
best_model_state_dict = None

val_accuracies = []

# 修改训练循环中的验证部分
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, warmup_scheduler)
    else:
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, cosine_scheduler)

    val_loss, val_accuracy, conf_matrix = validate(model, val_loader, criterion)
    val_accuracies.append(val_accuracy)

    # # 如果当前的验证准确率超过之前的最佳值，则保存模型
    # if val_accuracy > best_val_accuracy:
    #     best_val_accuracy = val_accuracy
    #     best_model_state_dict = model.state_dict()  # 保存最佳模型的状态字典
    #     torch.save(best_model_state_dict, f'best_model_{Net}_sub_{sub}.pth')  # 保存模型
    #
    # # 记录日志：训练和验证损失、准确率，及混淆矩阵
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # logger.info(f"Confusion Matrix:\n{conf_matrix}")

    # print(f"Epoch {epoch + 1}/{num_epochs}")
    # print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    # print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # print(f"Confusion Matrix:\n{conf_matrix}")

# 计算并记录平均验证准确率和最大验证准确率
avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
max_val_accuracy = max(val_accuracies)

logger.info(f"Average Val Accuracy: {avg_val_accuracy:.4f}")
logger.info(f"Max Val Accuracy: {max_val_accuracy:.4f}")

print(f"Average Val Accuracy: {avg_val_accuracy:.4f}")
print(f"Max Val Accuracy: {max_val_accuracy:.4f}")

