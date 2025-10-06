from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader,Subset
from torch.optim import Adam
import torch.nn as nn
import logging
from utils import BCI2aDataset
import random
import numpy as np
from models.Must_modefied import Must
# from models.EEGNet import EEGNet
# from models.conformer import Conformer
# from models.Must_woPW import Must
from models.EEGViT import EEGViT
# from models.EEGDeformer import Deformer
from models.EEGSwin import SwinTransformer1D

Net = 'Must'
import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--sub', type=int, required=True, help='Subject identifier')
args = parser.parse_args()

sub = args.sub

logging.basicConfig(filename='results/MI/'+Net+'/training_log_' + str(sub) + '.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数
batch_size = 72
learning_rate = 2e-4
weight_decay = 0
num_epochs = 2000
warmup_epochs = 0  # Warm-up阶段的epoch数
n_classes = 4

seed_n = 1234
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

training_file_path = r'/disk2/zhaokui/data/standard_2a_data/A0' + str(sub) + 'T.mat'
testing_file_path = r'/disk2/zhaokui/data/standard_2a_data/A0' + str(sub) + 'E.mat'

train_dataset = BCI2aDataset(training_file_path)
val_dataset = BCI2aDataset(testing_file_path)

train_loader = DataLoader(train_dataset, batch_size=72, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=72, shuffle=False)

# model = SwinTransformer1D(num_classes=4,in_chans=22,embed_dim=40,depths=[2,2,2],num_heads=[2,4,8],window_size=5,mlp_ratio=4).to(device)
model = Must(emb_size=100).to(device)
# model = EEGNet(nb_classes=3,Chans=62,Samples=200).to(device)
# model = Conformer().to(device)
# model = EEGViT(num_chan=22, num_time=1000, num_patches=40, num_classes=4).to(device)
# model = Deformer(num_chan=22, num_time=1000, temporal_kernel=11, num_kernel=40,
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


def interaug(timg, label):
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = torch.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        tmp_aug_data = torch.zeros(len(tmp_data), 1, 22, 1000)
        for ri in range(len(tmp_data)):
            for rj in range(8):
                rand_idx = torch.randint(0, tmp_data.shape[0], (8,))
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                  rj * 125:(rj + 1) * 125]
        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:len(tmp_data)])
    aug_data = torch.cat(aug_data)
    aug_label = torch.cat(aug_label)
    aug_shuffle = torch.randperm(len(aug_data))
    aug_data = aug_data[aug_shuffle, ...]
    aug_label = aug_label[aug_shuffle]

    return aug_data, aug_label


def train(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    lens = 0.0
    for data, labels in loader:
        data = data.unsqueeze(1)
        # data augmentation
        aug_data, aug_label = interaug(data, labels)

        # concat real train dataset and generate artificial train dataset
        data = torch.cat((data, aug_data))
        labels = torch.cat((labels, aug_label))

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


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_predictions = []

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
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)

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

    # 如果当前的验证准确率超过之前的最佳值，则保存模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state_dict = model.state_dict()  # 保存最佳模型的状态字典
        torch.save(best_model_state_dict, f'best_model_{Net}_sub_{sub}.pth')  # 保存模型

    # 记录日志：训练和验证损失、准确率，及混淆矩阵
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

# 计算并记录平均验证准确率和最大验证准确率
avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
max_val_accuracy = max(val_accuracies)

logger.info(f"Average Val Accuracy: {avg_val_accuracy:.4f}")
logger.info(f"Max Val Accuracy: {max_val_accuracy:.4f}")

print(f"Average Val Accuracy: {avg_val_accuracy:.4f}")
print(f"Max Val Accuracy: {max_val_accuracy:.4f}")

# # 如果需要在训练完成后保存最终模型的权重
# torch.save(model.state_dict(), f'results/MI/'+Net+'/final_model_{Net}_sub_{sub}.pth')
