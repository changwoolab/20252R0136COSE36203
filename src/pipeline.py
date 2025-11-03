"""
Main Pipeline for Korean Food Explanation System
Integrates classification, knowledge retrieval, and text generation
"""
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import json

from .classifier import KoreanFoodClassifier, create_food_classifier
from .cnn_classifier import CNNFoodClassifier, create_cnn_classifier
from .vit_classifier import ViTFoodClassifier, create_vit_classifier
from .knowledge_base import FoodKnowledgeBase
from .text_generator import create_explainer, SimpleFoodExplainer, FoodExplainer


class KoreanFoodPipeline:
    """
    Complete pipeline for Korean food image analysis and explanation
    
    Pipeline steps:
    1. Classify Korean food from image using CLIP
    2. Retrieve food information from knowledge base
    3. Generate natural language explanation using LLM
    """
    
    def __init__(
        self,
        knowledge_base_path: str,
        classifier_type: str = "clip",  # 'clip', 'cnn', or 'vit'
        clip_model: str = "openai/clip-vit-base-patch32",
        cnn_model_type: str = "resnet50",
        cnn_model_path: str = None,
        vit_model_type: str = "vit_base_patch16_224",
        vit_model_path: str = None,
        use_llm: bool = False,  # Set to False by default for faster inference
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = None
    ):
        """
        Initialize the pipeline
        
        Args:
            knowledge_base_path: Path to the food knowledge base JSON
            classifier_type: Type of classifier to use ('clip', 'cnn', or 'vit')
            clip_model: CLIP model name (if using CLIP)
            cnn_model_type: CNN architecture (if using CNN)
            cnn_model_path: Path to trained CNN model (if using CNN)
            vit_model_type: ViT architecture (if using ViT)
            vit_model_path: Path to trained ViT model (if using ViT)
            use_llm: Whether to use LLM for generation (slower but more natural)
            llm_model: LLM model name
            device: Device to run on
        """
        print("Initializing Korean Food Explanation Pipeline...")
        
        # Load knowledge base
        print("\n[1/3] Loading knowledge base...")
        self.knowledge_base = FoodKnowledgeBase(knowledge_base_path)
        
        # Get list of food names
        food_names = self.knowledge_base.get_food_names()
        print(f"Loaded {len(food_names)} food categories")
        
        # Initialize classifier based on type
        self.classifier_type = classifier_type.lower()
        
        if self.classifier_type == 'clip':
            print("\n[2/3] Loading CLIP classifier...")
            self.classifier = create_food_classifier(food_names, model_name=clip_model)
        
        elif self.classifier_type == 'cnn':
            print("\n[2/3] Loading CNN classifier...")
            self.classifier = create_cnn_classifier(
                food_names, 
                model_type=cnn_model_type,
                model_path=cnn_model_path
            )
            if cnn_model_path:
                print(f"Loaded trained CNN model from {cnn_model_path}")
            else:
                print(f"Using pretrained {cnn_model_type} (needs training on Korean food)")
        
        elif self.classifier_type == 'vit':
            print("\n[2/3] Loading ViT classifier...")
            try:
                self.classifier = create_vit_classifier(
                    food_names,
                    model_type=vit_model_type,
                    model_path=vit_model_path
                )
                if vit_model_path:
                    print(f"Loaded trained ViT model from {vit_model_path}")
                else:
                    print(f"Using pretrained {vit_model_type} (needs training on Korean food)")
            except ImportError as e:
                print(f"Error: {e}")
                print("Please install timm: pip install timm")
                raise
        
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}. Use 'clip', 'cnn', or 'vit'")
        
        # Initialize text generator
        print("\n[3/3] Loading text generator...")
        self.explainer = create_explainer(use_llm=use_llm, model_name=llm_model)
        
        print("\n✓ Pipeline initialized successfully!\n")
    
    def analyze_food_image(
        self, 
        image_path: str, 
        top_k: int = 3,
        confidence_threshold: float = 0.1
    ) -> Dict:
        """
        Complete pipeline: classify image and generate explanation
        
        Args:
            image_path: Path to the food image
            top_k: Number of top predictions to consider
            confidence_threshold: Minimum confidence for predictions
        
        Returns:
            Dictionary with classification results and explanation
        """
        # Step 1: Classify the image
        predictions = self.classifier.classify_image(image_path, top_k=top_k)
        
        # Filter by confidence
        predictions = [(name, conf) for name, conf in predictions if conf >= confidence_threshold]
        
        if not predictions:
            return {
                'success': False,
                'error': 'No confident predictions found',
                'predictions': []
            }
        
        # Get top prediction
        top_food_name, top_confidence = predictions[0]
        
        # Step 2: Retrieve food information
        food_info = self.knowledge_base.get_food_info(top_food_name)
        
        if not food_info:
            return {
                'success': False,
                'error': f'No information found for {top_food_name}',
                'predictions': predictions
            }
        
        # Step 3: Generate explanation
        explanation = self.explainer.generate_explanation(top_food_name, food_info)
        
        # Prepare result
        result = {
            'success': True,
            'identified_food': top_food_name,
            'korean_name': food_info['korean_name'],
            'confidence': top_confidence,
            'category': food_info['category'],
            'explanation': explanation,
            'predictions': predictions,
            'detailed_info': {
                'description': food_info['description'],
                'ingredients': food_info['ingredients'],
                'cooking_method': food_info['cooking_method'],
                'cultural_note': food_info['cultural_note']
            }
        }
        
        return result
    
    def analyze_batch(
        self, 
        image_paths: List[str], 
        confidence_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence
        
        Returns:
            List of results
        """
        results = []
        for image_path in image_paths:
            result = self.analyze_food_image(image_path, confidence_threshold=confidence_threshold)
            results.append(result)
        
        return results
    
    def get_food_explanation(self, food_name: str) -> Optional[str]:
        """
        Get explanation for a specific food by name
        
        Args:
            food_name: English name of the food
        
        Returns:
            Explanation text or None
        """
        food_info = self.knowledge_base.get_food_info(food_name)
        if not food_info:
            return None
        
        explanation = self.explainer.generate_explanation(food_name, food_info)
        return explanation
    
    def list_available_foods(self) -> List[str]:
        """Get list of all available food names"""
        return self.knowledge_base.get_food_names()
    
    def get_food_info(self, food_name: str) -> Optional[Dict]:
        """Get raw information about a food"""
        return self.knowledge_base.get_food_info(food_name)
    
    def format_result_text(self, result: Dict) -> str:
        """
        Format result as human-readable text
        
        Args:
            result: Result dictionary from analyze_food_image
        
        Returns:
            Formatted text
        """
        if not result['success']:
            return f"Error: {result.get('error', 'Unknown error')}"
        
        text = "=" * 60 + "\n"
        text += f"🍽️  Korean Food Identification Result\n"
        text += "=" * 60 + "\n\n"
        
        text += f"Identified Food: {result['identified_food']}\n"
        text += f"Korean Name: {result['korean_name']}\n"
        text += f"Confidence: {result['confidence']:.2%}\n"
        text += f"Category: {result['category']}\n"
        text += "\n" + "-" * 60 + "\n"
        text += f"📖 Explanation:\n"
        text += "-" * 60 + "\n"
        text += f"{result['explanation']}\n"
        
        if len(result['predictions']) > 1:
            text += "\n" + "-" * 60 + "\n"
            text += "Other possible matches:\n"
            for i, (name, conf) in enumerate(result['predictions'][1:], 2):
                text += f"  {i}. {name} ({conf:.2%})\n"
        
        text += "\n" + "=" * 60 + "\n"
        
        return text
    
    def format_result_json(self, result: Dict) -> str:
        """Format result as JSON"""
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def save_result(self, result: Dict, output_path: str, format: str = 'json'):
        """
        Save result to file
        
        Args:
            result: Result dictionary
            output_path: Path to save file
            format: 'json' or 'txt'
        """
        if format == 'json':
            content = self.format_result_json(result)
        else:
            content = self.format_result_text(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Result saved to {output_path}")


def create_pipeline(
    knowledge_base_path: str,
    classifier_type: str = "clip",
    use_llm: bool = False,
    **kwargs
) -> KoreanFoodPipeline:
    """
    Factory function to create a pipeline
    
    Args:
        knowledge_base_path: Path to knowledge base JSON
        classifier_type: Type of classifier ('clip', 'cnn', or 'vit')
        use_llm: Whether to use LLM for text generation
        **kwargs: Additional arguments for pipeline (cnn_model_path, vit_model_path, etc.)
    
    Returns:
        Initialized KoreanFoodPipeline
    """
    return KoreanFoodPipeline(
        knowledge_base_path=knowledge_base_path,
        classifier_type=classifier_type,
        use_llm=use_llm,
        **kwargs
    )

