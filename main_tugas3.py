
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from PIL import Image

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = (224, 224)

def get_device():
    """Get available device (GPU/MPS/CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device



# ============================================================================
# DATASET
# ============================================================================

class MakananIndo(Dataset):
    """Indonesian Food Dataset"""
    def __init__(self, data_dir='./train', img_size=(224, 224), split='train'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split

        # Load image files and labels
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg'))])
        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        label_dict = dict(zip(df['filename'], df['label']))

        # Filter valid data
        all_data = [(f, label_dict[f]) for f in self.image_files if f in label_dict]

        # Split dataset
        total_len = len(all_data)
        train_len = int(0.8 * total_len)
        indices = list(range(total_len))
        random.shuffle(indices)

        split_indices = indices[:train_len] if split == 'train' else indices[train_len:]
        self.data = [all_data[i] for i in split_indices]

        # Create label mapping
        labels_in_split = [label for _, label in self.data]
        unique_labels = sorted(set(labels_in_split))
        self.class_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx][0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = Image.fromarray(image)

        # Transform
        transform = Compose([
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        image = transform(image)

        label = self.class_to_index[self.data[idx][1]]
        return image, label, img_path


def create_dataloaders(batch_size_train=32, batch_size_val=64):
    """Create train and validation dataloaders"""
    train_dataset = MakananIndo(data_dir='./train', split='train', img_size=IMG_SIZE)
    val_dataset = MakananIndo(data_dir='./train', split='val', img_size=IMG_SIZE)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val,
                           shuffle=False, num_workers=num_workers)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {train_dataset.num_classes}")
    return train_loader, val_loader, train_dataset.class_to_index, train_dataset.num_classes


# ============================================================================
# MODEL BLOCKS
# ============================================================================

class PlainBlock(nn.Module):
    """Plain convolutional block without skip connections"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)


class Plain34(nn.Module):
    """Plain-34 Network: ResNet-34 architecture without residual connections"""
    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [PlainBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNetBlock(nn.Module):
    """ResNet block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out + identity)  # Skip connection
        return out


class ResNet34(nn.Module):
    """ResNet-34 Network with residual connections"""
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [ResNetBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate accuracy, F1, precision, and recall"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, f1, precision, recall


def train_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, num_classes)
    return avg_loss, accuracy, f1, precision, recall


def validate_epoch(model, val_loader, criterion, device, num_classes):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, num_classes)
    return avg_loss, accuracy, f1, precision, recall, all_labels, all_predictions


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def train_model(model, model_name, criterion, optimizer, epochs, learning_rate, batch_size_train, batch_size_val):
    """Main training function"""
    print(f"\n{'='*80}\nTRAINING {model_name}\n{'='*80}")

    # Setup
    device = get_device()
    model = model.to(device)

    # Create dataloaders
    train_loader, val_loader, class_to_idx, num_classes = create_dataloaders(batch_size_train, batch_size_val)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Learning rate: {learning_rate} | Epochs: {epochs}")
    print(f"Batch size: Train={batch_size_train}, Val={batch_size_val}\n")

    # Training loop
    best_val_accuracy = 0.0
    best_model_path = f"{model_name.lower()}_model.pth"
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        start_time = time.time()

        # Train and validate
        train_loss, train_acc, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_acc, val_f1, val_prec, val_rec, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device, num_classes)

        epoch_time = time.time() - start_time

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print results
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")

    # Final evaluation
    print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
    model.load_state_dict(torch.load(best_model_path))
    val_loss, val_acc, val_f1, val_prec, val_rec, val_labels, val_preds = validate_epoch(
        model, val_loader, criterion, device, num_classes)

    print(f"Best Validation Accuracy: {val_acc:.4f}")
    print(f"F1: {val_f1:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}\n")

    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, model_name)

    print(f"Model saved: {best_model_path}")
    return model, best_val_accuracy, (train_losses, train_accuracies, val_losses, val_accuracies)


def plot_training_history(train_losses, train_accs, val_losses, val_accs, model_name):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename = f'{model_name.lower()}_training_history.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training plot saved: {filename}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check dataset
    if not os.path.exists('train') or not os.path.exists('train.csv'):
        print("ERROR: 'train' folder or 'train.csv' not found!")
        print("Please ensure dataset is available in current directory.")
        exit(1)

    # Train Plain34
    # print("\n" + "="*80)
    # print("TRAINING PLAIN-34 (No Skip Connections)")
    # print("="*80)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # plain34_model = Plain34()
    # plain34_model, plain34_acc, plain34_history = train_model(plain34_model, "Plain34")

    # Train ResNet34
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 8

    print("\n" + "="*80)
    print("TRAINING RESNET-34 (With Skip Connections)")
    print("="*80)
    resnet34_model = ResNet34()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet34_model.parameters(), lr=LEARNING_RATE)
    resnet34_model, resnet34_acc, resnet34_history = train_model(resnet34_model, "ResNet34", criterion=criterion, optimizer=optimizer, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, batch_size_train=BATCH_SIZE_TRAIN, batch_size_val=BATCH_SIZE_VAL)

# Train ResNet34 With SGD With Nesterov Momentum
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VAL = 64
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 8

    print("\n" + "="*80)
    print("TRAINING RESNET-34 With AdamW (With Skip Connections, WD 5e-4)")
    print("="*80)
    resnet34_model_nesterov = ResNet34()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(resnet34_model_nesterov.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    resnet34_model, resnet34_acc, resnet34_history = train_model(resnet34_model, "ResNet34 - AdamW (WD 5e-4)", criterion=criterion, optimizer=optimizer, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, batch_size_train=BATCH_SIZE_TRAIN, batch_size_val=BATCH_SIZE_VAL)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """
    ResNeXt bottleneck: 1x1 reduce -> 3x3 grouped conv -> 1x1 expand, skip connection.
    Cardinality = jumlah grup (jalur paralel).
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        base_width = 4  # sederhana: tetap 4 seperti ResNeXt-50 32x4d

        D = int(math.floor(out_channels * (base_width / 64.0)))
        width = D * cardinality
        assert width % cardinality == 0 and width > 0, "width tidak valid untuk cardinality"

        # 1x1 reduce ke 'width'
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 grouped conv (multi-path implisit), stride di sini (ResNet v1.5)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1 expand ke out_channels * expansion
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = F.relu(out + identity)
        return out


class ResNeXt34(nn.Module):
    """ResNeXt gaya-34 (layout 3,4,6,3) dengan bottleneck ResNeXtBlock."""
    def __init__(self, num_classes=5, cardinality=32):
        super(ResNeXt34, self).__init__()
        self.cardinality = cardinality

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64,  64,  3, stride=1, cardinality=cardinality)
        self.stage2 = self._make_stage(64*ResNeXtBlock.expansion, 128, 4, stride=2, cardinality=cardinality)
        self.stage3 = self._make_stage(128*ResNeXtBlock.expansion, 256, 6, stride=2, cardinality=cardinality)
        self.stage4 = self._make_stage(256*ResNeXtBlock.expansion, 512, 3, stride=2, cardinality=cardinality)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, cardinality):
        downsample = None
        out_expanded = out_channels * ResNeXtBlock.expansion
        if stride != 1 or in_channels != out_expanded:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_expanded, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_expanded),
            )

        layers = [ResNeXtBlock(in_channels, out_channels, stride, downsample, cardinality=cardinality)]
        in_channels = out_expanded
        for _ in range(1, num_blocks):
            layers.append(ResNeXtBlock(in_channels, out_channels, cardinality=cardinality))
            in_channels = out_expanded
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Train ResNeXt34
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 8

print("\n" + "="*80)
print("TRAINING RESNEXT-34 (With Grouped Convolutions)")
print("="*80)

resnext34_model = ResNeXt34(cardinality=16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnext34_model.parameters(), lr=LEARNING_RATE)

resnext34_model, resnext34_acc, resnext34_history = train_model(
    resnext34_model,
    "ResNeXt34",
    criterion=criterion,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=BATCH_SIZE_VAL
)

# Train ResNeXt34 - SGD Nesterov 0.9
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 8

print("\n" + "="*80)
print("TRAINING RESNEXT-34 (AdamW, wd  5e-4)")
print("="*80)
resnext34_model = ResNeXt34(cardinality=16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(resnext34_model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

resnext34_model, resnext34_acc, resnext34_history = train_model(
    resnext34_model,
    "ResNeXt34 - AdamW (WD 5e-4)",
    criterion=criterion,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size_train=BATCH_SIZE_TRAIN,
    batch_size_val=BATCH_SIZE_VAL
)