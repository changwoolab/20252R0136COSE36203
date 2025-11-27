"""
Main Pipeline for Korean Food Explanation System
Integrates classification, knowledge retrieval, and text generation
"""
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import json
import torch

from .classifier import KoreanFoodClassifier, create_food_classifier
from .cnn_classifier import CNNFoodClassifier, create_cnn_classifier
from .vit_classifier import ViTFoodClassifier, create_vit_classifier
from .knowledge_base import FoodKnowledgeBase
from .attribute_db import AttributeDatabase
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
        clip_model_path: str = None,
        cnn_model_type: str = "resnet50",
        cnn_model_path: str = None,
        vit_model_type: str = "vit_base_patch16_224",
        vit_model_path: str = None,
        use_llm: bool = False,  # Set to False by default for faster inference
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        attribute_db_path: str = None,  # Path to attribute database
        device: str = None
    ):
        """
        Initialize the pipeline
        
        Args:
            knowledge_base_path: Path to the food knowledge base JSON
            classifier_type: Type of classifier to use ('clip', 'cnn', or 'vit')
            clip_model: CLIP model name (if using CLIP, HuggingFace model name)
            clip_model_path: Path to local fine-tuned CLIP model (overrides clip_model)
            cnn_model_type: CNN architecture (if using CNN)
            cnn_model_path: Path to trained CNN model (if using CNN)
            vit_model_type: ViT architecture (if using ViT)
            vit_model_path: Path to trained ViT model (if using ViT)
            use_llm: Whether to use LLM for generation (slower but more natural)
            llm_model: LLM model name
            attribute_db_path: Path to attribute database JSON
            device: Device to run on
        """
        print("Initializing Korean Food Explanation Pipeline...")
        
        # Set classifier type first (needed for any conditional logic)
        self.classifier_type = classifier_type.lower()
        
        # Load knowledge base
        print("\n[1/4] Loading knowledge base...")
        self.knowledge_base = FoodKnowledgeBase(knowledge_base_path)
        
        # Load attribute database
        print("\n[2/4] Loading attribute database...")
        if attribute_db_path:
            self.attribute_db = AttributeDatabase(attribute_db_path)
        else:
            # Use default attributes
            self.attribute_db = AttributeDatabase()
        print(f"Loaded {len(self.attribute_db.get_all_attributes())} attributes")
        
        # Get list of food names
        food_names = self.knowledge_base.get_food_names()
        print(f"Loaded {len(food_names)} food categories")
        
        # Initialize classifier based on type
        
        if self.classifier_type == 'clip':
            print("\n[3/4] Loading CLIP classifier...")
            if clip_model_path:
                # Load fine-tuned model
                from .classifier import KoreanFoodClassifier
                self.classifier = KoreanFoodClassifier(
                    model_name=clip_model,
                    model_path=clip_model_path
                )
                # Set food classes if not already loaded
                if not self.classifier.food_classes:
                    self.classifier.set_food_classes(food_names)
            else:
                self.classifier = create_food_classifier(food_names, model_name=clip_model)
        
        elif self.classifier_type == 'cnn':
            print("\n[3/4] Loading CNN classifier...")
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
            print("\n[3/4] Loading ViT classifier...")
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
        print("\n[4/4] Loading text generator...")
        self.explainer = create_explainer(
            use_llm=use_llm,
            model_name=llm_model
        )
        
        print("\n✓ Pipeline initialized successfully!\n")
    
    def _attribute_aware_retrieval(
        self,
        image_path: str,
        predicted_label: str,
        label_confidence: float,
        attribute_weight: float = 0.5
    ) -> Tuple[str, Dict]:
        """
        Attribute-aware retrieval: rank knowledge base entries by weighted combination of
        label similarity and image-attribute similarity.
        
        Args:
            image_path: Path to the food image
            predicted_label: Predicted food name from classifier
            label_confidence: Confidence score of the prediction
            attribute_weight: Weight for attribute similarity (0-1). 
                             Higher = more emphasis on attributes vs label match.
        
        Returns:
            Tuple of (best_match_food_name, food_info_dict)
        """
        if self.classifier_type != 'clip' or not self.knowledge_base:
            # Fallback to simple retrieval
            food_info = self.knowledge_base.get_food_info(predicted_label)
            return predicted_label, food_info or {}
        
        # Get image embedding
        image_features = self.classifier.get_image_embedding(image_path)
        image_tensor = torch.from_numpy(image_features).to(self.classifier.device)
        
        # Get all foods with attributes
        all_foods = self.knowledge_base.get_all_foods()
        food_attributes = self.knowledge_base.get_all_foods_with_attributes()
        
        scores = []
        
        for food_name, food_info in all_foods.items():
            # Score 1: Label similarity (exact match = 1.0, otherwise based on string similarity)
            if food_name.lower() == predicted_label.lower():
                label_score = 1.0
            else:
                # Simple string similarity (can be improved with fuzzy matching)
                label_score = 0.3 if predicted_label.lower() in food_name.lower() or food_name.lower() in predicted_label.lower() else 0.0
            
            # Score 2: Image-attribute similarity using CLIP
            attribute_string = food_attributes.get(food_name, "")
            if attribute_string:
                # Compute attribute text embedding
                attr_embedding = self.classifier.compute_text_embedding(attribute_string)
                # Compute cosine similarity
                attr_similarity = (image_tensor @ attr_embedding.T).item()
                # Normalize to [0, 1] range (cosine similarity is [-1, 1])
                attr_score = (attr_similarity + 1.0) / 2.0
            else:
                attr_score = 0.0
            
            # Weighted combination
            combined_score = (1.0 - attribute_weight) * label_score + attribute_weight * attr_score
            
            scores.append((food_name, combined_score, food_info))
        
        # Sort by combined score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match
        best_food_name, best_score, best_info = scores[0]
        
        return best_food_name, best_info
    
    def analyze_food_image(
        self, 
        image_path: str, 
        top_k: int = 3,
        confidence_threshold: float = None,
        candidate_foods: List[str] = None,
        use_attribute_retrieval: bool = True,
        attribute_weight: float = 0.5,
        use_attribute_db: bool = True  # Enable Strategy 2: Attribute Retrieval
    ) -> Dict:
        """
        Complete pipeline: classify image and generate explanation
        Implements Strategy 2 (Attribute Retrieval):
        1. Try exact match in DB (O(1))
        2. If fail, trigger Attribute Matching using CLIP
        3. Construct prompt with CLIP label and attributes
        
        Args:
            image_path: Path to the food image
            top_k: Number of top predictions to consider
            confidence_threshold: Minimum confidence for predictions (None = auto-detect)
            candidate_foods: List of food names for zero-shot classification (CLIP only)
            use_attribute_retrieval: Use attribute-aware retrieval for better matching
            attribute_weight: Weight for attribute similarity in retrieval (0-1)
            use_attribute_db: Enable attribute database retrieval for unknown foods
        
        Returns:
            Dictionary with classification results and explanation
        """
        # Auto-set confidence threshold based on classifier type
        if confidence_threshold is None:
            if self.classifier_type == 'clip':
                confidence_threshold = 0.001  # CLIP outputs very low probabilities
            else:
                confidence_threshold = 0.01  # CNN/ViT have higher confidence
        
        # Step 1: Primary CLIP - Predict food name
        if self.classifier_type == 'clip' and candidate_foods is not None:
            # Zero-shot classification: combine candidate foods with KB classes
            # This evaluates the model's ability to:
            # 1. Classify in-distribution classes (150 KB classes) correctly
            # 2. Generalize to unseen classes (10 zero-shot candidates) in zero-shot manner
            kb_food_names = self.knowledge_base.get_food_names() if self.knowledge_base else []
            
            # Combine candidate foods with KB classes (remove duplicates)
            combined_foods = []
            seen = set()
            
            # Add candidate foods first
            for food in candidate_foods:
                if food not in seen:
                    combined_foods.append(food)
                    seen.add(food)
            
            # Add KB classes that aren't already in candidate foods
            kb_added = 0
            for food in kb_food_names:
                if food not in seen:
                    combined_foods.append(food)
                    seen.add(food)
                    kb_added += 1
            
            print(f"[Zero-Shot Classification] {len(candidate_foods)} zero-shot candidates + {kb_added} KB classes = {len(combined_foods)} total classes")
            
            # Use zero-shot classification with combined list
            # Computes text features on-the-fly for all classes
            predictions = self.classifier.classify_image_zero_shot(
                image_path, 
                candidate_foods=combined_foods,
                top_k=top_k
            )
        else:
            # Standard classification (uses precomputed text features from set_food_classes)
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
        
        # Check if this is a zero-shot prediction (food not in KB but in candidate_foods)
        is_zero_shot_prediction = False
        if candidate_foods is not None and top_food_name in candidate_foods:
            # This is a zero-shot candidate - check if it's in KB
            kb_food_names = self.knowledge_base.get_food_names() if self.knowledge_base else []
            if top_food_name not in kb_food_names:
                is_zero_shot_prediction = True
                print(f"[Zero-Shot Result] '{top_food_name}' is a zero-shot prediction (not in KB)")
        
        # Strategy 2: Try exact match first (O(1))
        food_info = self.knowledge_base.get_food_info(top_food_name) if self.knowledge_base else None
        matched_food_name = top_food_name
        attribute_retrieval_used = False
        predicted_attributes = []
        attribute_descriptions = {}
        
        # For zero-shot predictions, create synthetic food_info with relevant KB information
        if is_zero_shot_prediction and not food_info:
            # Zero-shot food - retrieve relevant info from similar KB foods
            # Look at other predictions to find similar foods in KB
            similar_kb_foods = []
            kb_food_names = self.knowledge_base.get_food_names() if self.knowledge_base else []
            
            for pred_name, pred_conf in predictions[1:]:  # Skip first (the zero-shot prediction)
                if pred_name in kb_food_names:
                    similar_info = self.knowledge_base.get_food_info(pred_name)
                    if similar_info:
                        similar_kb_foods.append({
                            'name': pred_name,
                            'confidence': pred_conf,
                            'info': similar_info
                        })
                        if len(similar_kb_foods) >= 3:  # Get top 3 similar KB foods
                            break
            
            # Build context from similar KB foods for LLM
            similar_context = ""
            similar_categories = set()
            similar_ingredients = []
            
            if similar_kb_foods:
                similar_context = "Similar dishes from the knowledge base:\n"
                for similar in similar_kb_foods:
                    info = similar['info']
                    similar_context += f"\n- {info.get('english_name', similar['name'])} ({info.get('korean_name', '')}): "
                    similar_context += f"{info.get('description', '')[:200]}..."
                    if info.get('category'):
                        similar_categories.add(info.get('category'))
                    if info.get('ingredients'):
                        similar_ingredients.extend(info.get('ingredients', [])[:3])
            
            # Infer category from similar foods if possible
            inferred_category = list(similar_categories)[0] if len(similar_categories) == 1 else 'Unknown (Zero-Shot)'
            
            print(f"[Zero-Shot] Found {len(similar_kb_foods)} similar KB foods for context")
            
            food_info = {
                'english_name': top_food_name,
                'korean_name': '',
                'description': f'{top_food_name} is a traditional Korean dish.',
                'category': inferred_category,
                'ingredients': list(set(similar_ingredients))[:5],  # Unique similar ingredients
                'cooking_method': '',
                'cultural_note': '',
                'is_zero_shot': True,
                'similar_kb_foods': similar_kb_foods,
                'similar_context': similar_context
            }
            attribute_retrieval_used = False
        # If exact match fails and attribute DB is enabled, use attribute retrieval (but NOT for zero-shot)
        elif not food_info and use_attribute_db and self.classifier_type == 'clip' and hasattr(self, 'attribute_db'):
            # Step 2: Secondary CLIP - Predict attributes
            attribute_list = self.attribute_db.get_attribute_list()
            predicted_attributes = self.classifier.predict_attributes(
                image_path, 
                attribute_list, 
                top_k=5
            )
            
            # Extract attribute names
            attribute_names = [attr[0] for attr in predicted_attributes if attr[1] > 0.1]  # Filter by confidence
            
            if attribute_names:
                # Step 3: Retrieve attribute descriptions from Attribute DB
                attribute_descriptions = self.attribute_db.get_attributes(attribute_names)
                attribute_retrieval_used = True
                
                # Step 4: Generate explanation using attributes
                # Create a synthetic food_info dict with attributes
                food_info = {
                    'english_name': top_food_name,
                    'korean_name': '',
                    'description': '',  # Will be generated from attributes
                    'category': '',
                    'ingredients': [],
                    'cooking_method': '',
                    'cultural_note': '',
                    'attributes': attribute_descriptions,
                    'predicted_attributes': attribute_names
                }
        
        # If still no food_info, try attribute-aware retrieval as fallback (but NOT for zero-shot predictions)
        if not food_info and not is_zero_shot_prediction and use_attribute_retrieval and self.knowledge_base and self.classifier_type == 'clip':
            # Use attribute-aware retrieval for better matching
            matched_food_name, food_info = self._attribute_aware_retrieval(
                image_path,
                top_food_name,
                top_confidence,
                attribute_weight=attribute_weight
            )
        
        if not food_info:
            return {
                'success': False,
                'error': f'No information found for {matched_food_name}',
                'predictions': predictions,
                'predicted_attributes': [attr[0] for attr in predicted_attributes] if predicted_attributes else []
            }
        
        # Step 4: Generate explanation (with attributes if available)
        if attribute_retrieval_used and attribute_descriptions:
            # Use attribute-based generation (LLM or template)
            explanation = self.explainer.generate_explanation_from_attributes(
                matched_food_name, 
                attribute_descriptions,
                predicted_attributes
            )
        elif is_zero_shot_prediction and food_info.get('is_zero_shot'):
            # Zero-shot prediction - simple prompt with only food name + similar dishes (no attributes)
            # Check if the explainer has the zero-shot method (FoodExplainer with LLM)
            if hasattr(self.explainer, 'generate_zeroshot_explanation'):
                explanation = self.explainer.generate_zeroshot_explanation(matched_food_name, food_info)
            else:
                # Fallback to simple template for SimpleFoodExplainer (only food name + similar dishes)
                similar_kb_foods = food_info.get('similar_kb_foods', [])
                similar_names = [f['name'] for f in similar_kb_foods] if similar_kb_foods else []
                
                explanation = f"{matched_food_name} is a traditional Korean dish. "
                if similar_names:
                    explanation += f"It shares similarities with {', '.join(similar_names[:2])}."
        else:
            # Standard generation for in-distribution foods (LLM or template with attributes)
            # Get predicted attributes for in-distribution foods
            attr_names = [attr[0] for attr in predicted_attributes] if predicted_attributes else None
            
            # Check if explainer supports predicted_attributes parameter
            if hasattr(self.explainer, 'generate_explanation'):
                import inspect
                sig = inspect.signature(self.explainer.generate_explanation)
                if 'predicted_attributes' in sig.parameters:
                    explanation = self.explainer.generate_explanation(matched_food_name, food_info, predicted_attributes=attr_names)
                else:
                    explanation = self.explainer.generate_explanation(matched_food_name, food_info)
            else:
                explanation = self.explainer.generate_explanation(matched_food_name, food_info)
        
        # Prepare result
        result = {
            'success': True,
            'identified_food': matched_food_name,
            'predicted_food': top_food_name,  # Original prediction
            'korean_name': food_info.get('korean_name', ''),
            'confidence': top_confidence,
            'category': food_info.get('category', ''),
            'explanation': explanation,
            'predictions': predictions,
            'zero_shot': candidate_foods is not None,
            'attribute_retrieval_used': attribute_retrieval_used or (use_attribute_retrieval and self.classifier_type == 'clip'),
            'predicted_attributes': [attr[0] for attr in predicted_attributes] if predicted_attributes else [],
            'attribute_descriptions': attribute_descriptions,
            'detailed_info': {
                'description': food_info.get('description', ''),
                'ingredients': food_info.get('ingredients', []),
                'cooking_method': food_info.get('cooking_method', ''),
                'cultural_note': food_info.get('cultural_note', '')
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
        text += f"📖 Explanation (with Classifier-Retrieval):\n"
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
    clip_model: str = "openai/clip-vit-base-patch32",
    clip_model_path: str = None,
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    attribute_db_path: str = None,
    **kwargs
) -> KoreanFoodPipeline:
    """
    Factory function to create a pipeline
    
    Args:
        knowledge_base_path: Path to knowledge base JSON
        classifier_type: Type of classifier ('clip', 'cnn', or 'vit')
        use_llm: Whether to use LLM for text generation
        clip_model: CLIP model name (HuggingFace model name)
        clip_model_path: Path to local fine-tuned CLIP model
        llm_model: LLM model name (if use_llm is True)
        attribute_db_path: Path to attribute database JSON
        **kwargs: Additional arguments for pipeline (cnn_model_path, vit_model_path, etc.)
    
    Returns:
        Initialized KoreanFoodPipeline
    """
    return KoreanFoodPipeline(
        knowledge_base_path=knowledge_base_path,
        classifier_type=classifier_type,
        use_llm=use_llm,
        clip_model=clip_model,
        clip_model_path=clip_model_path,
        llm_model=llm_model,
        attribute_db_path=attribute_db_path,
        **kwargs
    )

