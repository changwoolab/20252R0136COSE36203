"""
Training script for fine-tuning CLIP on Korean food dataset
(Optional - CLIP works well in zero-shot mode too)
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import KoreanFoodClassifier
import config


class KoreanFoodDataset(Dataset):
    """Dataset for Korean food images"""
    
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        
        # Build dataset
        self.samples = []
        self.classes = []
        
        for food_dir in sorted(self.dataset_dir.iterdir()):
            if food_dir.is_dir():
                class_name = food_dir.name
                self.classes.append(class_name)
                class_idx = len(self.classes) - 1
                
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
        
        return image, class_idx, class_name


def create_data_loaders(dataset_dir, batch_size=32, train_split=0.8):
    """Create training and validation data loaders"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    dataset = KoreanFoodDataset(dataset_dir, transform=transform)
    
    # Split into train and val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader, dataset.classes


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model.model.get_image_features(pixel_values=images)
        
        # Compute similarity with text features
        text_features = model.text_features
        similarity = outputs @ text_features.T
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = similarity.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model"""
    model.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model.model.get_image_features(pixel_values=images)
            
            # Compute similarity
            text_features = model.text_features
            similarity = outputs @ text_features.T
            
            # Predictions
            _, predicted = similarity.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CLIP classifier on Korean food')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models/clip_finetuned', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training CLIP Classifier for Korean Food")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATASET_DIR):
        print(f"Error: Dataset not found at {config.DATASET_DIR}")
        sys.exit(1)
    
    # Create data loaders
    print("\nPreparing data...")
    train_loader, val_loader, classes = create_data_loaders(
        config.DATASET_DIR,
        batch_size=args.batch_size
    )
    
    # Initialize model
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KoreanFoodClassifier(device=device)
    model.set_food_classes(classes)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr
    )
    
    # Training loop
    print("\nStarting training...\n")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_acc = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_model(args.output_dir)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

