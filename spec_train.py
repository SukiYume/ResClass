import os
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_color_codes()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from timm.scheduler import CosineLRScheduler
from confusion_matrix import compute_confusion_matrix


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet1D(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool1d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SpecDataset(Dataset):

    def __init__(self, data_path='./Data/train_data_10.fits', val=False):
        with fits.open(data_path) as f:
            self.data  = f[0].data
            self.label = f[1].data['label']
        self.val       = val

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data  = torch.tensor(self.data[idx].astype(np.float32)).unsqueeze(0)
        data  = (data - data.mean()) / data.std()
        if not self.val:
            data += torch.randn_like(data) * data.std() / 3
            if np.random.choice([True, False]):
                change_pos = np.random.randint(0, data.shape[1])
                data[0, change_pos] += torch.abs(data).max() if np.random.choice([True, False]) else -torch.abs(data).max()
        label = self.label[idx]
        return data, label


def make_balanced_sampler(labels):
    class_weights = 1. / np.bincount(labels)
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


if __name__== '__main__':

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size                 = 64
    model_name                 = 'best_model.pth'
    model                      = ResNet1D(BasicBlock, [2, 2, 2, 2], num_classes=3).to(device)
    dataset                    = SpecDataset('./Data/train_data_10.fits')

    save_path                  = './logs/spec/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    num_epochs                 = 50
    train_size                 = int(0.8 * len(dataset))
    val_size                   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    sampler                    = make_balanced_sampler(train_dataset.dataset.label[train_dataset.indices])

    train_dataset.val          = False
    val_dataset.val            = True
    train_loader               = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader                 = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial=num_epochs//2, cycle_decay=0.5, cycle_limit=2, lr_min=1e-6, warmup_t=4, warmup_lr_init=1e-5)

    logs_train, logs_val, loss_min = [], [], pow(10, 10)
    for epoch in range(num_epochs):
        print('Epoch {}/{} with LR {:.6f}'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr']))
        with tqdm(total=len(train_loader), dynamic_ncols=True, ascii=True, desc='Epoch {}/{}'.format(epoch + 1, num_epochs), bar_format    = '{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
            model.train()
            train_loss, train_acc = [], []
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).sum().item() / labels.size(0)
                train_acc.append(acc)
                train_loss.append(loss.item())

                pbar.set_postfix({'loss': loss.item(), 'acc': acc})
                pbar.update()

        with tqdm(total=len(val_loader), dynamic_ncols=True, ascii=True, desc='Val Epoch {}/{}'.format(epoch + 1, num_epochs), bar_format    = '{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
            model.eval()
            val_loss, val_acc = [], []
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    acc = (predicted == labels).sum().item() / labels.size(0)
                    val_acc.append(acc)
                    val_loss.append(loss.item())

                    pbar.set_postfix({'loss': loss.item(), 'acc': acc})
                    pbar.update()

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%\n'.format(
                epoch+1, num_epochs, np.mean(train_loss), 100 * np.mean(train_acc), np.mean(val_loss), 100 * np.mean(val_acc)
        ))
        val_epoch_loss = np.mean(val_loss)
        if val_epoch_loss <= loss_min:
            loss_min = val_epoch_loss
            print('loss is min')
            print('save model...')
            torch.save(model.state_dict(), save_path + model_name)
            print('done\n')

            print('plot confusion matrix\n')
            confusion_mat = compute_confusion_matrix(model, val_loader, device)
            ax = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Purples',xticklabels=np.arange(3).astype(np.int64), yticklabels=np.arange(3).astype(np.int64))
            for _, spine in ax.spines.items():
                spine.set_visible(True)
            plt.ylabel('Ground Truth')
            plt.xlabel('Predicted')
            plt.savefig('{}/matrix.png'.format(save_path), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print('loss is not min\n')

        scheduler.step(epoch + 1)
        logs_train.append([np.mean(train_loss), 100 * np.mean(train_acc)])
        logs_val.append([np.mean(val_loss), 100 * np.mean(val_acc)])

    log_data = np.stack([logs_train, logs_val], axis=1)
    np.save('{}/logs.npy'.format(save_path), log_data)