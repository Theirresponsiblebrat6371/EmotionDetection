import torch
import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
torch.backends.cudnn.benchmark = True


# Image transformations (FER images are 48x48 grayscale)
train_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Paths
train_dir = "archive/train"
test_dir = "archive/test"

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

num_classes = len(train_dataset.classes)


model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)



for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True




targets = train_dataset.targets
class_counts = Counter(targets)

weights = torch.tensor(
    [1.0 / class_counts[i] for i in range(len(class_counts))],
    device=device
)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)


optimizer = torch.optim.AdamW([
    {"params": model.layer3.parameters(), "lr": 5e-5},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 3e-4},
])



EPOCHS = 30 

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} done")



# Testing the model
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

print("Test Accuracy:", 100 * correct / total)

torch.save({
    "model_state": model.state_dict(),
    "classes": train_dataset.classes
}, "emotion_resnet.pth")
