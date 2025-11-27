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
import json


class KoreanFoodClassifier:
    """CLIP-based classifier for Korean food images"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None, model_path: str = None):
        """
        Initialize the classifier
        
        Args:
            model_name: HuggingFace model name for CLIP (used if model_path is None)
            device: Device to run on ('cuda' or 'cpu')
            model_path: Path to local fine-tuned model directory (overrides model_name)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned CLIP model from {model_path}")
            self.model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
            
            # Load class mappings if available
            class_mapping_path = os.path.join(model_path, 'class_mappings.json')
            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    mappings = json.load(f)
                    if 'food_classes' in mappings:
                        self.food_classes = mappings['food_classes']
                        print(f"Loaded {len(self.food_classes)} food classes from saved model")
        else:
            if model_path:
                print(f"Warning: Model path {model_path} not found, using {model_name}")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        self.food_classes = getattr(self, 'food_classes', [])
        self.text_features = None
        if self.food_classes:
            self._compute_text_features()
        
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
        # IMPROVED: Use multiple prompt templates and ensemble them for better accuracy
        prompt_templates = [
            "a photo of {food}",
            "a picture of {food}",
            "an image of {food}",
            "{food}",
            "Korean food {food}",
            "a dish of {food}"
        ]
        
        all_text_features = []
        
        # Compute features for each prompt template
        for template in prompt_templates:
            text_prompts = [template.format(food=food) for food in self.food_classes]
            
            # Tokenize
            inputs = self.tokenizer(
                text_prompts, 
                padding=True, 
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            # Get text features
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_text_features.append(text_features)
        
        # Average the features from all templates (ensemble)
        self.text_features = torch.stack(all_text_features).mean(dim=0)
        # Re-normalize after averaging
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        
        print(f"Computed text features: {self.text_features.shape} (using {len(prompt_templates)} prompt templates)")
    
    def _compute_text_features_for_foods(self, food_names: List[str]) -> torch.Tensor:
        """
        Compute text embeddings for arbitrary food names (zero-shot)
        
        Args:
            food_names: List of food names to compute features for
        
        Returns:
            Normalized text feature tensor of shape (len(food_names), feature_dim)
        """
        print(f"[Zero-Shot] Computing text features on-the-fly for {len(food_names)} food classes...")
        
        # Use multiple prompt templates and ensemble them for better accuracy
        prompt_templates = [
            "a photo of {food}",
            "a picture of {food}",
            "an image of {food}",
            "{food}",
            "Korean food {food}",
            "a dish of {food}"
        ]
        
        all_text_features = []
        
        # Compute features for each prompt template
        for template in prompt_templates:
            text_prompts = [template.format(food=food) for food in food_names]
            
            # Tokenize
            inputs = self.tokenizer(
                text_prompts, 
                padding=True, 
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            # Get text features
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_text_features.append(text_features)
        
        # Average the features from all templates (ensemble)
        text_features = torch.stack(all_text_features).mean(dim=0)
        # Re-normalize after averaging
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
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
        
        # IMPROVED: Use temperature scaling for better probability distribution
        temperature = 0.1  # Lower temperature = sharper distribution
        scaled_similarity = similarity / temperature
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(scaled_similarity, dim=0)
        
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
        
        # IMPROVED: Use temperature scaling for better probability distribution
        temperature = 0.1
        scaled_similarity = similarity / temperature
        
        # Apply softmax
        probs = torch.nn.functional.softmax(scaled_similarity, dim=1)
        
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
        """Save the model and class mappings"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save class mappings
        if self.food_classes:
            class_mappings = {
                'food_classes': self.food_classes,
                'num_classes': len(self.food_classes),
                'model_type': 'clip'
            }
            with open(os.path.join(save_path, 'class_mappings.json'), 'w') as f:
                import json
                json.dump(class_mappings, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"Model loaded from {model_path}")
        
        # Recompute text features if food classes are set
        if self.food_classes:
            self._compute_text_features()
    
    def ensemble_with_zeroshot(self, zero_shot_model_name: str, alpha: float):
        """
        Apply WiSE-FT (Weight-Space Ensembling) by interpolating with zero-shot model
        
        Formula: θ_final = α · θ_fine-tuned + (1 - α) · θ_zero-shot
        
        Args:
            zero_shot_model_name: HuggingFace model name for the original zero-shot CLIP model
            alpha: Ensemble weight (0-1). Higher alpha = more weight on fine-tuned model.
                   alpha=1.0 = pure fine-tuned, alpha=0.0 = pure zero-shot
        """
        print(f"\nApplying WiSE-FT with α={alpha:.3f}...")
        print(f"Loading zero-shot model: {zero_shot_model_name}")
        
        # Load zero-shot model
        zero_shot_model = CLIPModel.from_pretrained(zero_shot_model_name).to(self.device)
        
        # Get state dicts
        fine_tuned_state = self.model.state_dict()
        zero_shot_state = zero_shot_model.state_dict()
        
        # Ensure both models have the same architecture
        assert set(fine_tuned_state.keys()) == set(zero_shot_state.keys()), \
            "Model architectures must match for ensembling"
        
        # Perform weight-space ensembling
        ensembled_state = {}
        for key in fine_tuned_state.keys():
            fine_tuned_weight = fine_tuned_state[key]
            zero_shot_weight = zero_shot_state[key]
            
            # Linear interpolation: θ_final = α · θ_fine-tuned + (1 - α) · θ_zero-shot
            ensembled_weight = alpha * fine_tuned_weight + (1 - alpha) * zero_shot_weight
            ensembled_state[key] = ensembled_weight
        
        # Load ensembled weights into model
        self.model.load_state_dict(ensembled_state)
        
        print(f"✓ Weight-space ensembling complete")
        print(f"  Fine-tuned weight: {alpha:.1%}")
        print(f"  Zero-shot weight: {(1-alpha):.1%}")
        
        # Recompute text features with ensembled model
        if self.food_classes:
            self._compute_text_features()
        
        # Clean up zero-shot model
        del zero_shot_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    
    def classify_image_zero_shot(
        self, 
        image_path: str, 
        candidate_foods: List[str] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Zero-shot classification: classify image against arbitrary food names
        
        This method enables true zero-shot capability - you can classify against
        any food names without precomputing features or being limited to a fixed set.
        
        Args:
            image_path: Path to the image file
            candidate_foods: List of food names to classify against. If None, uses
                           the precomputed food_classes (backward compatibility)
            top_k: Number of top predictions to return
        
        Returns:
            List of (food_name, confidence) tuples
        """
        # If no candidate foods provided, fall back to precomputed features
        if candidate_foods is None:
            if self.text_features is None or len(self.food_classes) == 0:
                raise ValueError("No candidate foods provided and no precomputed food classes. "
                               "Either provide candidate_foods or call set_food_classes() first.")
            return self.classify_image(image_path, top_k=top_k)
        
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
        
        # Compute text features on-the-fly for candidate foods
        text_features = self._compute_text_features_for_foods(candidate_foods)
        
        # Compute similarity
        similarity = (image_features @ text_features.T).squeeze(0)
        
        # Use temperature scaling for better probability distribution
        temperature = 0.1  # Lower temperature = sharper distribution
        scaled_similarity = similarity / temperature
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(scaled_similarity, dim=0)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(candidate_foods)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            food_name = candidate_foods[idx.item()]
            confidence = prob.item()
            results.append((food_name, confidence))
        
        return results
    
    def compute_text_embedding(self, text: str) -> torch.Tensor:
        """
        Compute text embedding for a single text string using CLIP
        
        Args:
            text: Text string to encode
        
        Returns:
            Normalized text feature tensor
        """
        inputs = self.tokenizer(
            [text],
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict_attributes(self, image_path: str, attribute_list: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict attributes from an image using CLIP (Secondary CLIP for attributes)
        
        Args:
            image_path: Path to the food image
            attribute_list: List of attribute names to predict (e.g., ["Spicy", "Grilled", "Chicken"])
            top_k: Number of top attributes to return
        
        Returns:
            List of (attribute_name, confidence_score) tuples, sorted by confidence
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute text features for all attributes
        text_inputs = self.tokenizer(
            attribute_list,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        similarity = (image_features @ text_features.T).squeeze(0)
        
        # Apply temperature scaling for sharper distribution
        temperature = 0.1
        scaled_similarity = similarity / temperature
        probs = torch.nn.functional.softmax(scaled_similarity, dim=0)
        
        # Get top-k attributes
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(attribute_list)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            attribute_name = attribute_list[idx.item()]
            confidence = prob.item()
            results.append((attribute_name, confidence))
        
        return results


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

