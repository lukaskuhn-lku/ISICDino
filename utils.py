import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier as KNN
from tqdm import tqdm

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.

    Parameters
    ----------
    model : nn.Module
        Module.

    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def compute_knn_accuracy(backbone, knn_train_dataset, knn_val_dataset, k=200):
    backbone.eval()
    knn_train_loader = DataLoader(knn_train_dataset, batch_size=128, shuffle=False)
    knn_val_loader = DataLoader(knn_val_dataset, batch_size=128, shuffle=False)

    train_features = []
    train_labels = []
    for images, labels in knn_train_loader:
        features = backbone(images)
        train_features.append(features)
        train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    val_features = []
    val_labels = []
    for images, labels in knn_val_loader:
        features = backbone(images)
        val_features.append(features)
        val_labels.append(labels)

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    knn = KNN(n_neighbors=k)
    knn.fit(train_features, train_labels)
    val_predictions = knn.predict(val_features)
    accuracy = (val_predictions == val_labels).float().mean().item()

    return accuracy

def compute_knn_accuracy(backbone, knn_train_dataset, knn_val_dataset, device='cuda', k=200):
    backbone.eval()
    backbone.to(device)
    knn_train_loader = DataLoader(knn_train_dataset, batch_size=16, shuffle=False, num_workers=4)
    knn_val_loader = DataLoader(knn_val_dataset, batch_size=16, shuffle=False, num_workers=4)

    train_features = []
    train_labels = []
    for images, labels in tqdm(knn_train_loader):
        images = images.to(device)
        features = backbone(images)
        train_features.append(features.cpu().detach())
        train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    val_features = []
    val_labels = []
    for images, labels in tqdm(knn_val_loader):
        images = images.to(device)
        features = backbone(images)
        val_features.append(features.cpu().detach())
        val_labels.append(labels)

    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    knn = KNN(n_neighbors=k)
    knn.fit(train_features, train_labels)
    val_predictions = knn.predict(val_features)
    accuracy = (val_predictions == val_labels).to(float).mean().item()

    backbone.train()
    return accuracy, val_predictions, train_labels, val_labels