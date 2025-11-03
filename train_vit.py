"""
Training script for ViT-based Korean Food Classifier
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vit_classifier import ViTFoodClassifier
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


def create_data_loaders(dataset_dir, batch_size=32, train_split=0.8, img_size=224):
    """Create training and validation data loaders"""
    
    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model.model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ViT classifier on Korean food')
    parser.add_argument('--model-type', type=str, default='vit_base_patch16_224',
                        choices=['vit_tiny_patch16_224', 'vit_small_patch16_224', 
                                'vit_base_patch16_224', 'vit_large_patch16_224'],
                        help='ViT model type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models/vit_trained', 
                        help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Training ViT Classifier ({args.model_type}) for Korean Food")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATASET_DIR):
        print(f"Error: Dataset not found at {config.DATASET_DIR}")
        sys.exit(1)
    
    # Determine image size from model type
    img_size = 384 if '384' in args.model_type else 224
    
    # Create data loaders
    print("\nPreparing data...")
    train_loader, val_loader, classes = create_data_loaders(
        config.DATASET_DIR,
        batch_size=args.batch_size,
        img_size=img_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = ViTFoodClassifier(model_type=args.model_type, num_classes=len(classes), device=device)
        model.set_food_classes(classes)
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease install timm: pip install timm")
        sys.exit(1)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        model.load_model(args.resume)
        # Try to load training state
        checkpoint_file = os.path.join(args.resume, 'training_state.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                training_state = json.load(f)
                start_epoch = training_state.get('epoch', 0) + 1
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\nStarting training...\n")
    best_val_acc = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
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
                'val_acc': val_acc
            }
            with open(os.path.join(args.output_dir, 'training_state.json'), 'w') as f:
                json.dump(training_state, f, indent=2)
            
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()



