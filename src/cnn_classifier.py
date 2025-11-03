"""
CNN-based Korean Food Classifier
Uses ResNet50 or other CNN architectures for classification
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import os
import json


class CNNFoodClassifier:
    """CNN-based classifier for Korean food images"""
    
    def __init__(self, model_type: str = "resnet50", num_classes: int = 150, device: str = None):
        """
        Initialize the CNN classifier
        
        Args:
            model_type: Type of CNN model ('resnet50', 'resnet101', 'efficientnet_b0')
            num_classes: Number of food categories
            device: Device to run on ('cuda' or 'cpu')
        """
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
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Initialized {model_type} CNN classifier")
    
    def _create_model(self, model_type: str, num_classes: int) -> nn.Module:
        """Create CNN model architecture with pretrained weights"""
        
        if model_type == 'resnet50':
            # Use modern weights API (not deprecated pretrained=True)
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Replace final layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained ResNet50 (ImageNet weights)")
        
        elif model_type == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained ResNet101 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B0 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B1 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b2':
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B2 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B3 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B4 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b5':
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B5 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b6':
            model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B6 (ImageNet weights)")
        
        elif model_type == 'efficientnet_b7':
            model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained EfficientNet-B7 (ImageNet weights)")
        
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            print(f"✓ Loaded pretrained MobileNet-V2 (ImageNet weights)")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier (for transfer learning)"""
        if self.model_type.startswith('resnet'):
            # Freeze all layers except fc
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print("✓ Froze backbone layers (only training final layer)")
        
        elif self.model_type.startswith('efficientnet') or self.model_type.startswith('mobilenet'):
            # Freeze all layers except classifier
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
            print("✓ Froze backbone layers (only training final layer)")
    
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
        """Get image embedding from the second-to-last layer"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get features before final classification layer
        with torch.no_grad():
            if self.model_type.startswith('resnet'):
                # For ResNet, extract features before fc layer
                x = self.model.conv1(image_tensor)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                features = torch.flatten(x, 1)
            else:
                # For other models, use forward hook (simplified)
                features = self.model.features(image_tensor)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
        
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


def create_cnn_classifier(
    food_names: List[str], 
    model_type: str = "resnet50",
    model_path: str = None
) -> CNNFoodClassifier:
    """
    Helper function to create and initialize a CNN classifier
    
    Args:
        food_names: List of food class names
        model_type: Type of CNN model
        model_path: Path to saved model (optional)
    
    Returns:
        Initialized CNNFoodClassifier
    """
    classifier = CNNFoodClassifier(model_type=model_type, num_classes=len(food_names))
    classifier.set_food_classes(food_names)
    
    if model_path and os.path.exists(model_path):
        classifier.load_model(model_path)
    
    return classifier



