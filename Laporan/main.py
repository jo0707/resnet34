import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import os
import time
import random
import zipfile
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from typing import Tuple, Optional, Dict, List
from torchvision import transforms
from PIL import Image

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Global variables
class_to_index = {}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample layer for dimension matching (if needed)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample to identity if needed (for dimension matching)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = F.relu(out)
        return out

class Plain34(nn.Module):
    """
    Plain-34 Network: ResNet-34 architecture without residual connections.

    Architecture:
    - Initial conv layer (7x7, stride=2)
    - MaxPool (3x3, stride=2)
    - 4 stages of Plain blocks:
      - Stage 1: 3 blocks, 64 channels
      - Stage 2: 4 blocks, 128 channels, stride=2 for first block
      - Stage 3: 6 blocks, 256 channels, stride=2 for first block
      - Stage 4: 3 blocks, 512 channels, stride=2 for first block
    - Global Average Pool
    - Fully Connected layer
    """
    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Plain block stages
        self.stage1 = self._make_stage(64, 64, 3, stride=1)    # 3 blocks, 64 channels
        self.stage2 = self._make_stage(64, 128, 4, stride=2)   # 4 blocks, 128 channels
        self.stage3 = self._make_stage(128, 256, 6, stride=2)  # 6 blocks, 256 channels
        self.stage4 = self._make_stage(256, 512, 3, stride=2)  # 3 blocks, 512 channels

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage consisting of multiple PlainBlocks.

        Args:
            in_channels: Input channels for the first block
            out_channels: Output channels for all blocks in this stage
            num_blocks: Number of blocks in this stage
            stride: Stride for the first block (usually 1 or 2)
        """
        downsample = None

        # If we need to change dimensions or stride, create downsample layer
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []

        # First block (may have stride=2 and different input/output channels)
        layers.append(PlainBlock(in_channels, out_channels, stride, downsample))

        # Remaining blocks (stride=1, same input/output channels)
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
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
        # Initial conv + bn + relu + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Plain block stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNetBlock(nn.Module):
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        # PENAMBAHAN SKIP CONNECTIIONS (hasil layer + input)
        out = F.relu(out + identity)

        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1)    # 3 blocks, 64 channels
        self.stage2 = self._make_stage(64, 128, 4, stride=2)   # 4 blocks, 128 channels
        self.stage3 = self._make_stage(128, 256, 6, stride=2)  # 6 blocks, 256 channels
        self.stage4 = self._make_stage(256, 512, 3, stride=2)  # 3 blocks, 512 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))

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
        # Initial conv + bn + relu + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Plain block stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# LOAD DATASET
class MakananIndo(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 data_dir='./train',
                 img_size=(224, 224),
                 transform=None,
                 split='train',
                 ):

        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.split = split

        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
        self.image_files.sort()

        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        self.label_dict = dict(zip(df['filename'], df['label']))
        self.labels = [self.label_dict.get(f, None) for f in self.image_files]
        all_data = list(zip(self.image_files, self.labels))

        all_data = [item for item in all_data if item[1] is not None]
        total_len = len(all_data)
        train_len = int(0.8 * total_len)

        indices = list(range(total_len))
        random.shuffle(indices)  # This uses the global random seed set above

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        if split == 'train':
            self.data = [all_data[i] for i in train_indices]
        elif split == 'val':
            self.data = [all_data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'")

        labels_in_split = [label for _, label in self.data]
        unique_labels = sorted(list(set(labels_in_split)))
        self.class_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        global class_to_index
        class_to_index = self.class_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx][0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        image = Image.fromarray(image)

        image = self.default_transform(image)

        # Get the string label and convert it to integer index
        string_label = self.data[idx][1]
        label = self.class_to_index[string_label]

        # return image data, label, and file_path
        return_data = (image, label, img_path)

        return return_data

    def default_transform(self, image):
        transform = Compose([
            ToTensor(),
            Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        return transform(image)

def create_label_encoder(dataset):
    all_labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        all_labels.append(label)

    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}

    return label_to_idx, idx_to_label, unique_labels

def calculate_metrics(y_true, y_pred, num_classes):
    accuracy = accuracy_score(y_true, y_pred)

    # For multiclass classification, use 'weighted' average for imbalanced datasets
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, f1, precision, recall

def train_epoch(model, train_loader, criterion, optimizer, device, label_to_idx):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, batch_data in enumerate(train_loader):
        images, labels, filepath = batch_data
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

        if batch_idx % 20 == 0:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {curr_lr:.2e}')

    avg_loss = running_loss / len(train_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))

    return avg_loss, accuracy, f1, precision, recall

def validate_epoch(model, val_loader, criterion, device, label_to_idx):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data in val_loader:
            images, labels, _ = batch_data
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))

    return avg_loss, accuracy, f1, precision, recall, all_labels, all_predictions

def train_model(model, model_name):
    print("="*80)
    print(f"TRAINING {model_name} ON INDONESIAN FOOD DATASET")
    print("="*80)

    # Device setup
    device = check_set_gpu()

    print('Using dataset from local train folder')

    # Minimal transforms - no augmentation for baseline Plain-34
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Create datasets
    train_dataset = MakananIndo(data_dir='./train', split='train', img_size=(224, 224), transform=train_transform)
    val_dataset = MakananIndo(data_dir='./train', split='val', img_size=(224, 224), transform=val_transform)

    # Create label encoder
    label_to_idx, idx_to_label, unique_labels = create_label_encoder(train_dataset)
    num_classes = len(unique_labels)

    print(f"Train data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")

    # Hyperparameters
    train_batch_size = 32  # Smaller batch size for Plain-34
    val_batch_size = 64
    learning_rate = 1e-3   # Standard learning rate
    num_epochs = 8        # More epochs for baseline training

    # Create data loaders
    cpu_count = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=min(4, cpu_count))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=min(4, cpu_count))

    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function only - using basic SGD without scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(f"\nStarting training with:")
    print(f"- Device: {device}")
    print(f"- Image size: 224x224")
    print(f"- Train Batch size: {train_batch_size}")
    print(f"- Validation Batch size: {val_batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Number of epochs: {num_epochs}")
    print("-" * 80)

    # Training loop
    best_val_accuracy = 0.0
    best_model_path = f"{model_name.lower()}_model.pth"

    # Storage for metrics
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)

        start_time = time.time()

        # Training phase
        train_loss, train_acc, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, device, label_to_idx
        )

        # Validation phase
        val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device, label_to_idx
        )

        epoch_time = time.time() - start_time

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")

        print("-" * 80)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Final evaluation with detailed classification report
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS - PLAIN-34 BASELINE")
    print("="*80)

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Final validation evaluation
    val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
        model, val_loader, criterion, device, label_to_idx
    )

    print(f"Final Validation Metrics:")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"Loss:      {val_loss:.4f}")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 50)
    class_names = [str(cls) for cls in unique_labels]
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Plain-34 Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Plain-34 Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs+1), [vl - tl for tl, vl in zip(train_losses, val_losses)], 'g-', label='Val - Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Monitor (Val Loss - Train Loss)')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('plain34_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining history plot saved as 'plain34_training_history.png'")
    print(f"Best model saved as: {best_model_path}")

    return model, best_val_accuracy, (train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    import sys

    # Check if train folder exists
    if not os.path.exists('train'):
        print("Dataset folder 'train' not found in current directory.")
        print("Please ensure the 'train' folder with images and train.csv exists.")

    if not os.path.exists('train.csv'):
        print("train.csv not found in current directory.")
        print("Please ensure train.csv exists alongside the 'train' folder.")

    plain34model = Plain34()
    model, best_accuracy, history = train_model(plain34model, "Plain34")

    resnet34model = ResNet34()
    model, best_accuracy, history = train_model(resnet34model, "ResNet34")

    print(f"\nTraining completed successfully!")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    print("\n" + "="*50)
    print("MODEL ARCHITECTURE TEST COMPLETED!")
    print("="*50)
    print("To train the model on Indonesian food dataset, run:")
    print("python plain34.py --train")
    print("\nNext steps:")
    print("1. Ensure dataset.zip is available (Indonesian food dataset)")
    print("2. Run training with: python plain34.py --train")
    print("3. Compare results with ResNet-34 (with residual connections)")
    print("4. Analyze the impact of skip connections on performance")