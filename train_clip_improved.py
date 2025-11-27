"""
Improved Training script for CLIP-based Korean Food Classifier
WITH ANTI-OVERFITTING TECHNIQUES AND PROPER CLIP FINE-TUNING
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
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])
    
    # Simple validation transform
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
    full_dataset = KoreanFoodDataset(dataset_dir, transform=None)
    
    # Split into train and val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
    
    # Apply transforms
    class TransformDataset:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
    
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 4,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes


def clip_contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Compute CLIP-style contrastive loss
    
    Args:
        image_features: (batch_size, feature_dim)
        text_features: (num_classes, feature_dim)
        temperature: Temperature scaling parameter
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix: (batch_size, num_classes)
    logits = (image_features @ text_features.T) / temperature
    
    # Create labels (each image should match its corresponding text)
    # Assuming we have labels for each image
    return logits


def train_epoch(model, train_loader, optimizer, device, epoch, scheduler=None, 
                freeze_vision=False, freeze_text=False, max_grad_norm=1.0):
    """Train for one epoch"""
    model.model.train()
    
    # Freeze/unfreeze components
    if freeze_vision:
        for param in model.model.vision_model.parameters():
            param.requires_grad = False
    else:
        for param in model.model.vision_model.parameters():
            param.requires_grad = True
    
    if freeze_text:
        for param in model.model.text_model.parameters():
            param.requires_grad = False
    else:
        for param in model.model.text_model.parameters():
            param.requires_grad = True
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass - get image features
        image_features = model.model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get text features (cached)
        text_features = model.text_features  # (num_classes, feature_dim)
        
        # Compute similarity matrix
        # Use temperature scaling for training
        temperature = 0.07
        logits = (image_features @ text_features.T) / temperature
        
        # Compute loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model"""
    model.model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            image_features = model.model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with text features
            text_features = model.text_features
            logits = image_features @ text_features.T
            
            # Predictions
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Compute top-5 accuracy
    with torch.no_grad():
        correct_top5 = 0
        total_top5 = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = model.model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = model.text_features
            logits = image_features @ text_features.T
            
            _, top5_pred = logits.topk(5, dim=1)
            total_top5 += labels.size(0)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    top5_accuracy = 100. * correct_top5 / total_top5
    
    return accuracy, top5_accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Training for CLIP Classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output-dir', type=str, default='./models/clip_improved', help='Output directory')
    parser.add_argument('--freeze-epochs', type=int, default=0, help='Number of epochs to freeze vision/text encoders')
    parser.add_argument('--freeze-vision', action='store_true', help='Freeze vision encoder initially')
    parser.add_argument('--freeze-text', action='store_true', help='Freeze text encoder initially')
    parser.add_argument('--warmup-epochs', type=int, default=2, help='Number of warmup epochs for LR')
    parser.add_argument('--early-stopping', type=int, default=5, help='Early stopping patience (0 to disable)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--ensemble', type=float, default=None, 
                        help='WiSE-FT ensemble weight (0-1). If provided, creates ensembled model: α·fine-tuned + (1-α)·zero-shot. Higher = more fine-tuned weight.')
    parser.add_argument('--zeroshot-model', type=str, default="openai/clip-vit-base-patch32",
                        help='Zero-shot model name for WiSE-FT (default: same as fine-tuned base model)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Improved CLIP Fine-tuning for Korean Food Classification")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Freeze epochs: {args.freeze_epochs}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Early stopping patience: {args.early_stopping}")
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
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(classes)}")
    
    # Initialize model
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Store original model name for WiSE-FT
    original_model_name = args.zeroshot_model
    
    model = KoreanFoodClassifier(device=device, model_name=original_model_name)
    model.set_food_classes(classes)
    
    # Check ensemble flag
    if args.ensemble is not None:
        if not (0.0 <= args.ensemble <= 1.0):
            print(f"Error: Ensemble weight must be between 0 and 1, got {args.ensemble}")
            sys.exit(1)
        print(f"\nWiSE-FT will be applied after training with α={args.ensemble:.3f}")
        print(f"  Formula: θ_final = {args.ensemble:.3f}·θ_fine-tuned + {1-args.ensemble:.3f}·θ_zero-shot")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("\nStarting training...\n")
    best_val_acc = 0
    best_val_top5 = 0
    patience_counter = 0
    train_history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # Determine if we should freeze components
        freeze_vision = args.freeze_vision and epoch < args.freeze_epochs
        freeze_text = args.freeze_text and epoch < args.freeze_epochs
        
        if freeze_vision or freeze_text:
            print(f"Freezing: vision={freeze_vision}, text={freeze_text}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scheduler=scheduler,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
            max_grad_norm=args.max_grad_norm
        )
        
        # Validate
        val_acc, val_top5 = validate(model, val_loader, device)
        
        # Update scheduler (if using ReduceLROnPlateau)
        # scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Val Top-5 Acc: {val_top5:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_top5': val_top5
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_top5 = val_top5
            patience_counter = 0
            
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_model(args.output_dir)
            
            # Save training history
            with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
                json.dump(train_history, f, indent=2)
            
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%, top-5: {val_top5:.2f}%)")
        else:
            patience_counter += 1
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation top-5 accuracy: {best_val_top5:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    
    # Apply WiSE-FT if requested
    if args.ensemble is not None:
        print("\n" + "=" * 70)
        print("Applying WiSE-FT (Weight-Space Ensembling)...")
        print("=" * 70)
        
        # Load the best model (it was already saved)
        # Apply ensembling
        model.ensemble_with_zeroshot(original_model_name, args.ensemble)
        
        # Save ensembled model
        ensemble_output_dir = args.output_dir + "_wiseft_alpha" + str(args.ensemble).replace('.', '_')
        os.makedirs(ensemble_output_dir, exist_ok=True)
        model.save_model(ensemble_output_dir)
        
        # Save ensemble info
        ensemble_info = {
            'ensemble_weight': args.ensemble,
            'zero_shot_model': original_model_name,
            'fine_tuned_model': args.output_dir,
            'formula': f"θ_final = {args.ensemble}·θ_fine-tuned + {1-args.ensemble}·θ_zero-shot"
        }
        with open(os.path.join(ensemble_output_dir, 'ensemble_info.json'), 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        print(f"\n✓ WiSE-FT model saved to: {ensemble_output_dir}")
        print(f"  This model preserves zero-shot capabilities while benefiting from fine-tuning")
    
    print("=" * 70)


if __name__ == "__main__":
    main()


