
# Question 5 - CIFAR-10 MLP Classifier

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

patch_size = 4
batch_size = 256
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def extract_patches(imgs, p=patch_size):
    # split image into patches
    n, c, h, w = imgs.shape
    x = imgs.unfold(2, p, p).unfold(3, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.view(n, -1)


train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.2),
])

eval_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfm)
full_train_eval = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=eval_tfm)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tfm)

# 90/10 split
val_size = int(0.1 * len(full_train))
train_size = len(full_train) - val_size
train_split, val_split = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(Subset(full_train_eval, val_split.indices), batch_size=batch_size, shuffle=False, pin_memory=True)
train_eval_loader = DataLoader(Subset(full_train_eval, train_split.indices), batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

input_size = (32 // patch_size) ** 2 * patch_size * patch_size * 3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = extract_patches(x)
        return self.net(x)


model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-5)

best_val_acc = 0.0
best_state = None

for ep in range(1, epochs + 1):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stop exploding gradients
        optimiser.step()
    scheduler.step()

    # check val accuracy
    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)
    val_acc = 100.0 * correct / seen

    print("epoch", ep, "/", epochs, "  val acc:", round(val_acc, 2))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

# reload best model
model.load_state_dict(best_state)
print("best val accuracy:", round(best_val_acc, 2))

model.eval()
correct = 0
seen = 0
with torch.no_grad():
    for imgs, labels in train_eval_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
train_acc = 100.0 * correct / seen


def test_mlp(model, loader=test_loader):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            pred_list.append(preds)
            label_list.append(labels.numpy())

    predicted_labels = np.concatenate(pred_list)
    true_labels = np.concatenate(label_list)
    accuracy = 100.0 * (predicted_labels == true_labels).sum() / len(true_labels)

    return predicted_labels, accuracy


predicted_labels, test_acc = test_mlp(model)

np.save("q5_predicted_labels.npy", predicted_labels)

print("train acc:", round(train_acc, 2))
print("val acc:", round(best_val_acc, 2))
print("test acc:", round(test_acc, 2))
print("predictions saved to q5_predicted_labels.npy")
