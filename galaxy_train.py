import os, h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_color_codes()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch, torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from timm.scheduler import CosineLRScheduler
from confusion_matrix import compute_confusion_matrix


class GalaxyDataset(Dataset):

    def __init__(self, data_path='./Data/Galaxy10.h5', val=False):

        with h5py.File(data_path, 'r') as file:
            self.data  = file['images'][10:]
            self.label = file['ans'][10:]

        self.val       = val
        self.trans     = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.9, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data  = self.transform(self.data[idx])
        label = self.label[idx]
        return data, label

    def transform(self, data):
        data = Image.fromarray(data.astype(np.uint8))
        if not self.val:
            data = self.trans(data)
        data  = transforms.ToTensor()(data)
        data  = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(data)
        return data


class GalaxyResNet(torch.nn.Module):

    def __init__(self, model_name='resnet18', num_classes=1):
        super(GalaxyResNet, self).__init__()
        model_dict = {
            'resnet18':  [torchvision.models.resnet18(weights=None), 512],
            'resnet50':  [torchvision.models.resnet50(weights=None), 2048]
        }
        basemodel, num_ch  = model_dict[model_name]
        self.base_model    = basemodel
        self.base_model.fc = torch.nn.Linear(num_ch, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x


def make_balanced_sampler(labels):
    class_weights = 1. / np.bincount(labels)
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler



if __name__ == '__main__':

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size                 = 64
    model                      = GalaxyResNet('resnet50', num_classes=10).to(device)
    dataset                    = GalaxyDataset('./Data/Galaxy10.h5')

    save_path                  = './logs/galaxy/'
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
    scheduler = CosineLRScheduler(optimizer, t_initial=num_epochs-5, cycle_decay=0.5, cycle_limit=1, lr_min=1e-6, warmup_t=4, warmup_lr_init=1e-5)

    logs_train, logs_val, loss_min = [], [], pow(10, 10)
    for epoch in range(num_epochs):
        print('Epoch {}/{} with LR {:.6f}'.format(epoch + 1, num_epochs, optimizer.param_groups[0]['lr']))
        with tqdm(total=len(train_loader), dynamic_ncols=True, ascii=True, desc='Epoch {}/{}'.format(epoch + 1, num_epochs), bar_format='{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
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

        with tqdm(total=len(val_loader), dynamic_ncols=True, ascii=True, desc='Val Epoch {}/{}'.format(epoch + 1, num_epochs), bar_format='{desc:16}{percentage:3.0f}%|{bar:20}{r_bar}') as pbar:
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

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%\n'.format(epoch+1, num_epochs, np.mean(train_loss), 100 * np.mean(train_acc), np.mean(val_loss), 100 * np.mean(val_acc)))
        val_epoch_loss = np.mean(val_loss)
        if val_epoch_loss <= loss_min:
            loss_min = val_epoch_loss
            print('loss is min')
            print('save model...')
            torch.save(model.state_dict(), save_path + 'best_model.pth')
            print('done\n')

            print('plot confusion matrix\n')
            confusion_mat = compute_confusion_matrix(model, val_loader, device)
            ax = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Purples',xticklabels=np.arange(10).astype(np.int64), yticklabels=np.arange(10).astype(np.int64))
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