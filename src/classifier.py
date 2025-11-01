"""
CLIP-based Korean Food Classifier
Uses OpenAI's CLIP model for zero-shot and fine-tuned classification
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import os


class KoreanFoodClassifier:
    """CLIP-based classifier for Korean food images"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize the classifier
        
        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        self.food_classes = []
        self.text_features = None
        
    def set_food_classes(self, food_names: List[str]):
        """
        Set the list of food classes to classify
        
        Args:
            food_names: List of English food names
        """
        self.food_classes = food_names
        print(f"Set {len(food_names)} food classes")
        
        # Precompute text embeddings for all food classes
        self._compute_text_features()
    
    def _compute_text_features(self):
        """Compute and cache text embeddings for all food classes"""
        # Create text prompts for each food
        text_prompts = [f"a photo of {food}" for food in self.food_classes]
        
        # Tokenize
        inputs = self.tokenizer(
            text_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.text_features = text_features
        print(f"Computed text features: {text_features.shape}")
    
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
        
        # Process image
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity with all text features
        similarity = (image_features @ self.text_features.T).squeeze(0)
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(similarity, dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.food_classes)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            food_name = self.food_classes[idx.item()]
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
        # Load images
        images = [Image.open(path).convert("RGB") for path in image_paths]
        
        # Process images
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = image_features @ self.text_features.T
        
        # Apply softmax
        probs = torch.nn.functional.softmax(similarity, dim=1)
        
        # Get top-k for each image
        results = []
        for i in range(len(image_paths)):
            top_probs, top_indices = torch.topk(probs[i], k=min(top_k, len(self.food_classes)))
            
            image_results = []
            for prob, idx in zip(top_probs, top_indices):
                food_name = self.food_classes[idx.item()]
                confidence = prob.item()
                image_results.append((food_name, confidence))
            
            results.append(image_results)
        
        return results
    
    def save_model(self, save_path: str):
        """Save the model"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"Model loaded from {model_path}")
        
        # Recompute text features if food classes are set
        if self.food_classes:
            self._compute_text_features()
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get image embedding vector"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
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


def create_food_classifier(food_names: List[str], model_name: str = "openai/clip-vit-base-patch32") -> KoreanFoodClassifier:
    """
    Helper function to create and initialize a food classifier
    
    Args:
        food_names: List of food class names
        model_name: CLIP model name
    
    Returns:
        Initialized KoreanFoodClassifier
    """
    classifier = KoreanFoodClassifier(model_name=model_name)
    classifier.set_food_classes(food_names)
    return classifier

