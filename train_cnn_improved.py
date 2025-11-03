"""
Improved Training script for CNN-based Korean Food Classifier
WITH ANTI-OVERFITTING TECHNIQUES
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cnn_classifier import CNNFoodClassifier
import config


class KoreanFoodDataset(Dataset):
    """Dataset for Korean food images"""
    
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        
        # Build dataset
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        for food_dir in sorted(self.dataset_dir.iterdir()):
            if food_dir.is_dir():
                class_name = food_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)
                
                class_idx = self.class_to_idx[class_name]
                
                # Get all images
                for img_path in food_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx, class_name))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx


def create_data_loaders(dataset_dir, batch_size=32, train_split=0.8):
    """Create training and validation data loaders with STRONG augmentation"""
    
    # IMPROVED: Much stronger data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Increased from 10
        transforms.RandomVerticalFlip(p=0.1),  # NEW: Vertical flip
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,
            saturation=0.3,
            hue=0.1  # NEW: Hue jitter
        ),
        transforms.RandomGrayscale(p=0.1),  # NEW: Random grayscale
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # NEW: Perspective transform
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW: Translation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # NEW: Random erasing
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    full_dataset = KoreanFoodDataset(dataset_dir, transform=train_transform)
    
    # Split into train and val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # NEW: Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes


# NEW: Label smoothing loss
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


# NEW: Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True, use_label_smoothing=True):
    """Train for one epoch with improved techniques"""
    model.model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply mixup
        if use_mixup and random.random() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            
            # Forward pass
            outputs = model.model(images)
            
            # Mixup loss
            if use_label_smoothing:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = lam * nn.CrossEntropyLoss()(outputs, labels_a) + (1 - lam) * nn.CrossEntropyLoss()(outputs, labels_b)
        else:
            # Forward pass (normal)
            outputs = model.model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model.model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        model.save_model(path)
        self.val_loss_min = val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN classifier with anti-overfitting techniques')
    parser.add_argument('--model-type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101',
                                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                                'mobilenet_v2'],
                        help='CNN model type')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--output-dir', type=str, default='./models/cnn_trained', 
                        help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no-mixup', action='store_true', help='Disable mixup augmentation')
    parser.add_argument('--no-label-smoothing', action='store_true', help='Disable label smoothing')
    parser.add_argument('--freeze-epochs', type=int, default=1, help='Number of epochs to train with frozen backbone (0 to disable)')
    parser.add_argument('--no-freeze', action='store_true', help='Disable freezing backbone (train all layers from start)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Training CNN Classifier ({args.model_type}) with Anti-Overfitting")
    print("=" * 70)
    print(f"\nRegularization techniques enabled:")
    print(f"  ✓ Pretrained backbone (ImageNet)")
    print(f"  ✓ Two-stage training: {not args.no_freeze} (freeze {args.freeze_epochs} epochs, then fine-tune)")
    print(f"  ✓ Strong data augmentation")
    print(f"  ✓ Weight decay (L2): {args.weight_decay}")
    print(f"  ✓ Mixup augmentation: {not args.no_mixup}")
    print(f"  ✓ Label smoothing: {not args.no_label_smoothing}")
    print(f"  ✓ Gradient clipping")
    print(f"  ✓ Early stopping (patience={args.patience})")
    print(f"  ✓ Cosine annealing scheduler")
    
    # Check if dataset exists
    if not os.path.exists(config.DATASET_DIR):
        print(f"\nError: Dataset not found at {config.DATASET_DIR}")
        sys.exit(1)
    
    # Create data loaders
    print("\nPreparing data with strong augmentation...")
    train_loader, val_loader, classes = create_data_loaders(
        config.DATASET_DIR,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNFoodClassifier(model_type=args.model_type, num_classes=len(classes), device=device)
    model.set_food_classes(classes)
    
    # IMPROVED: Freeze backbone for initial training (transfer learning best practice)
    if not args.no_freeze and args.freeze_epochs > 0:
        model.freeze_backbone()
        print(f"\n🔒 Stage 1: Training only final layer for {args.freeze_epochs} epochs")
        print("   (This prevents destroying pretrained features)")
        freeze_stage = True
    else:
        freeze_stage = False
        print("\n🔓 Training all layers from start")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        model.load_model(args.resume)
        checkpoint_file = os.path.join(args.resume, 'training_state.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                training_state = json.load(f)
                start_epoch = training_state.get('epoch', 0) + 1
    
    # IMPROVED: Use label smoothing loss
    if not args.no_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        print("Using label smoothing (0.1)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard cross entropy loss")
    
    # IMPROVED: Use AdamW with weight decay instead of Adam
    optimizer = optim.AdamW(
        model.model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # IMPROVED: Use cosine annealing instead of step LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # IMPROVED: Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    print("\nStarting training...\n")
    best_val_acc = 0
    
    for epoch in range(start_epoch, args.epochs):
        # IMPROVED: Unfreeze after freeze_epochs for fine-tuning
        if freeze_stage and epoch == args.freeze_epochs:
            print("\n" + "=" * 70)
            print(f"🔓 Stage 2: Unfreezing backbone for fine-tuning")
            print("=" * 70)
            model.unfreeze_backbone()
            
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
            print(f"   Reduced learning rate to {args.lr * 0.1:.6f} for fine-tuning\n")
            freeze_stage = False
        
        stage = "Stage 1 (Frozen)" if freeze_stage else "Stage 2 (Fine-tune)" if not args.no_freeze and args.freeze_epochs > 0 else "Training"
        print(f"\nEpoch {epoch+1}/{args.epochs} - {stage}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=not args.no_mixup,
            use_label_smoothing=not args.no_label_smoothing
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate overfitting gap
        gap = train_acc - val_acc
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Overfitting Gap: {gap:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_model(args.output_dir)
            
            # Save training state
            training_state = {
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'gap': gap
            }
            with open(os.path.join(args.output_dir, 'training_state.json'), 'w') as f:
                json.dump(training_state, f, indent=2)
            
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%, gap: {gap:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss, model, args.output_dir)
        if early_stopping.early_stop:
            print("\n⚠ Early stopping triggered!")
            break
    
    print("\n" + "=" * 70)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


