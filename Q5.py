import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

# settings
patch_size = 4
batch_size = 256
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# split each image into non-overlapping patches, then flatten and concatenate them
def extract_patches(imgs, p=patch_size):
    batch_size, channels, height, width = imgs.shape

    patches = imgs.reshape(
        batch_size,
        channels,
        height // p,
        p,
        width // p,
        p
    )

    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(batch_size, -1)

    return patches


# transforms for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.2),
])

# transforms for validation and test
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# load CIFAR-10
full_train = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

full_train_eval = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=eval_transform
)

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=eval_transform
)


# make 90/10 train/validation split
val_size = int(0.1 * len(full_train))
train_size = len(full_train) - val_size

train_split, val_split = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


# dataloaders
train_loader = DataLoader(
    train_split,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    Subset(full_train_eval, val_split.indices),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

train_eval_loader = DataLoader(
    Subset(full_train_eval, train_split.indices),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# input size after patching and flattening
input_size = (32 // patch_size) ** 2 * patch_size * patch_size * 3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = extract_patches(x)
        return self.net(x)


model = MLP().to(device)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=epochs,
    eta_min=1e-5
)

best_val_acc = 0.0
best_state = None


# training loop
for ep in range(1, epochs + 1):
    model.train()

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    scheduler.step()

    # validation accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total
    print("epoch", ep, "/", epochs, " val acc:", round(val_acc, 2))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())


# load best model
model.load_state_dict(best_state)
print("best val accuracy:", round(best_val_acc, 2))


# training accuracy on clean images
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_eval_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

train_acc = 100.0 * correct / total


def test_mlp(model, loader=test_loader):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()

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
