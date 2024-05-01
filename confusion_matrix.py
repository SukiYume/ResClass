import torch
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(model, dataloader, device):
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return confusion_matrix(all_targets, all_preds)
