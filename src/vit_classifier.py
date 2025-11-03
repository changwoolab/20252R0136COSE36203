"""
ViT-based Korean Food Classifier
Uses Vision Transformer (ViT) architecture for classification
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import os
import json

# Try to import timm for ViT models
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: 'timm' not installed. Install with: pip install timm")


class ViTFoodClassifier:
    """ViT-based classifier for Korean food images"""
    
    def __init__(self, model_type: str = "vit_base_patch16_224", num_classes: int = 150, device: str = None):
        """
        Initialize the ViT classifier
        
        Args:
            model_type: Type of ViT model
                - 'vit_tiny_patch16_224': Tiny ViT (5.7M params)
                - 'vit_small_patch16_224': Small ViT (22M params)
                - 'vit_base_patch16_224': Base ViT (86M params)
                - 'vit_large_patch16_224': Large ViT (304M params)
            num_classes: Number of food categories
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_TIMM:
            raise ImportError("Please install timm: pip install timm")
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.num_classes = num_classes
        self.food_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model(model_type, num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        # ViT typically uses 224x224 or 384x384
        img_size = 384 if '384' in model_type else 224
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Initialized {model_type} ViT classifier (image size: {img_size})")
    
    def _create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create ViT model architecture with pretrained weights"""
        
        try:
            # Create model with pretrained weights (ImageNet-21k or ImageNet-1k)
            model = timm.create_model(
                model_type,
                pretrained=True,
                num_classes=num_classes
            )
            print(f"✓ Loaded pretrained {model_type} (ImageNet weights)")
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_type}': {e}")
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier head (for transfer learning)"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the head (classifier)
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            for param in self.model.fc.parameters():
                param.requires_grad = True
        
        print("✓ Froze backbone layers (only training classifier head)")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ Unfroze all layers for fine-tuning")
    
    def set_food_classes(self, food_names: List[str]):
        """
        Set the list of food classes
        
        Args:
            food_names: List of English food names
        """
        self.food_classes = food_names
        self.num_classes = len(food_names)
        
        # Create mappings
        self.class_to_idx = {name: idx for idx, name in enumerate(food_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(food_names)}
        
        print(f"Set {len(food_names)} food classes")
    
    def classify_image(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify a single image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
        
        Returns:
            List of (food_name, confidence) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.food_classes)))
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            food_name = self.idx_to_class.get(idx.item(), f"Unknown_{idx.item()}")
            confidence = prob.item()
            results.append((food_name, confidence))
        
        return results
    
    def classify_batch(self, image_paths: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Classify multiple images in batch
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions per image
        
        Returns:
            List of prediction lists
        """
        # Load and preprocess images
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image_tensor = self.transform(image)
            images.append(image_tensor)
        
        # Stack into batch
        batch = torch.stack(images).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k for each image
        results = []
        for i in range(len(image_paths)):
            top_probs, top_indices = torch.topk(probabilities[i], k=min(top_k, len(self.food_classes)))
            
            image_results = []
            for prob, idx in zip(top_probs, top_indices):
                food_name = self.idx_to_class.get(idx.item(), f"Unknown_{idx.item()}")
                confidence = prob.item()
                image_results.append((food_name, confidence))
            
            results.append(image_results)
        
        return results
    
    def save_model(self, save_path: str):
        """Save the model and class mappings"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        model_file = os.path.join(save_path, 'model.pth')
        torch.save(self.model.state_dict(), model_file)
        
        # Save class mappings
        mappings = {
            'food_classes': self.food_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': {int(k): v for k, v in self.idx_to_class.items()},
            'model_type': self.model_type,
            'num_classes': self.num_classes
        }
        
        mappings_file = os.path.join(save_path, 'class_mappings.json')
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        # Load class mappings
        mappings_file = os.path.join(model_path, 'class_mappings.json')
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        
        self.food_classes = mappings['food_classes']
        self.class_to_idx = mappings['class_to_idx']
        self.idx_to_class = {int(k): v for k, v in mappings['idx_to_class'].items()}
        self.model_type = mappings['model_type']
        self.num_classes = mappings['num_classes']
        
        # Recreate model architecture
        self.model = self._create_model(self.model_type, self.num_classes)
        
        # Load weights
        model_file = os.path.join(model_path, 'model.pth')
        state_dict = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get image embedding from the model"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get features before final classification layer
        with torch.no_grad():
            # Forward through all layers except the final classifier
            features = self.model.forward_features(image_tensor)
            # Global average pooling
            if len(features.shape) == 3:  # [batch, seq_len, hidden_dim]
                # Use the class token embedding (first token) for ViT
                features = features[:, 0, :]
            else:  # Handle other formats
                features = features.mean(dim=1)
        
        return features.cpu().numpy()
    
    def evaluate(self, test_data: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate classifier on test data
        
        Args:
            test_data: Dictionary mapping food names to lists of image paths
        
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = 0
        top5_correct = 0
        
        for true_label, image_paths in test_data.items():
            for image_path in image_paths:
                predictions = self.classify_image(image_path, top_k=5)
                
                # Check top-1 accuracy
                pred_label = predictions[0][0]
                if pred_label == true_label:
                    correct += 1
                
                # Check top-5 accuracy
                top5_labels = [pred[0] for pred in predictions]
                if true_label in top5_labels:
                    top5_correct += 1
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        top5_accuracy = top5_correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'total_samples': total
        }


def create_vit_classifier(
    food_names: List[str], 
    model_type: str = "vit_base_patch16_224",
    model_path: str = None
) -> ViTFoodClassifier:
    """
    Helper function to create and initialize a ViT classifier
    
    Args:
        food_names: List of food class names
        model_type: Type of ViT model
        model_path: Path to saved model (optional)
    
    Returns:
        Initialized ViTFoodClassifier
    """
    classifier = ViTFoodClassifier(model_type=model_type, num_classes=len(food_names))
    classifier.set_food_classes(food_names)
    
    if model_path and os.path.exists(model_path):
        classifier.load_model(model_path)
    
    return classifier



