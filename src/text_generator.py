"""
Text Generator for Korean Food Explanations
Uses TinyLlama or similar small LLM to generate natural language explanations
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Optional
import json


class FoodExplainer:
    """Generates natural language explanations about Korean food"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = None):
        """
        Initialize the text generator
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading text generation model on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        # Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        print(f"Text generation model loaded successfully")
    
    def generate_explanation(
        self, 
        food_name: str, 
        food_info: Dict,
        max_length: int = 250,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a natural language explanation about a Korean food
        
        Args:
            food_name: English name of the food
            food_info: Dictionary with food information (from knowledge base)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
        
        Returns:
            Generated explanation text
        """
        # Create prompt with food information
        prompt = self._create_prompt(food_name, food_info)
        
        # Generate text
        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract the response part (after the prompt)
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            print(f"Error generating text: {e}")
            # Fallback to template-based generation
            return self._generate_template_explanation(food_name, food_info)
    
    def _create_prompt(self, food_name: str, food_info: Dict) -> str:
        """Create a prompt for the LLM"""
        
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        category = food_info.get('category', '')
        ingredients = food_info.get('ingredients', [])
        cooking_method = food_info.get('cooking_method', '')
        cultural_note = food_info.get('cultural_note', '')
        
        # Format ingredients
        ingredients_str = ", ".join(ingredients) if ingredients else "traditional Korean ingredients"
        
        # Create conversational prompt
        prompt = f"""<|system|>
You are a knowledgeable Korean food expert. Explain Korean dishes in a friendly, informative way.</s>
<|user|>
Tell me about {food_name} ({korean_name}).</s>
<|assistant|>
{food_name} ({korean_name}) is a {category.lower()}. {description}

The dish is made with {ingredients_str}. {cooking_method}

{cultural_note}</s>
"""
        
        return prompt
    
    def _generate_template_explanation(self, food_name: str, food_info: Dict) -> str:
        """Generate explanation using templates as fallback"""
        
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        category = food_info.get('category', '')
        ingredients = food_info.get('ingredients', [])
        cooking_method = food_info.get('cooking_method', '')
        cultural_note = food_info.get('cultural_note', '')
        
        # Build explanation
        explanation = f"{food_name} ({korean_name}) is a {category.lower()} in Korean cuisine. "
        explanation += f"{description} "
        
        if ingredients:
            ingredients_str = ", ".join(ingredients[:5])  # Limit to 5 ingredients
            explanation += f"It is typically made with {ingredients_str}. "
        
        if cooking_method:
            explanation += f"{cooking_method}. "
        
        if cultural_note:
            explanation += f"{cultural_note}"
        
        return explanation.strip()
    
    def generate_short_summary(self, food_name: str, food_info: Dict) -> str:
        """Generate a short one-sentence summary"""
        description = food_info.get('description', '')
        # Get first sentence
        first_sentence = description.split('.')[0] + '.'
        return first_sentence
    
    def generate_detailed_explanation(
        self, 
        food_name: str, 
        food_info: Dict,
        include_history: bool = True
    ) -> Dict[str, str]:
        """
        Generate a detailed structured explanation
        
        Returns:
            Dictionary with different sections of explanation
        """
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        category = food_info.get('category', '')
        ingredients = food_info.get('ingredients', [])
        cooking_method = food_info.get('cooking_method', '')
        cultural_note = food_info.get('cultural_note', '')
        
        result = {
            'title': f"{food_name} ({korean_name})",
            'category': category,
            'overview': description,
            'ingredients': ", ".join(ingredients) if ingredients else "Various traditional ingredients",
            'preparation': cooking_method,
            'cultural_significance': cultural_note
        }
        
        return result


class SimpleFoodExplainer:
    """Lightweight explainer that doesn't require LLM (template-based)"""
    
    def __init__(self):
        """Initialize simple explainer"""
        pass
    
    def generate_explanation(self, food_name: str, food_info: Dict) -> str:
        """Generate explanation using templates"""
        
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        category = food_info.get('category', '')
        ingredients = food_info.get('ingredients', [])
        cooking_method = food_info.get('cooking_method', '')
        cultural_note = food_info.get('cultural_note', '')
        
        # Build explanation
        explanation = f"**{food_name}** ({korean_name})\n\n"
        explanation += f"**Category:** {category}\n\n"
        explanation += f"**Description:** {description}\n\n"
        
        if ingredients:
            ingredients_str = ", ".join(ingredients)
            explanation += f"**Key Ingredients:** {ingredients_str}\n\n"
        
        if cooking_method:
            explanation += f"**Preparation:** {cooking_method}\n\n"
        
        if cultural_note:
            explanation += f"**Cultural Note:** {cultural_note}\n"
        
        return explanation.strip()
    
    def generate_short_summary(self, food_name: str, food_info: Dict) -> str:
        """Generate a short summary"""
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        first_sentence = description.split('.')[0] + '.'
        return f"{food_name} ({korean_name}): {first_sentence}"


def create_explainer(use_llm: bool = True, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> object:
    """
    Factory function to create an explainer
    
    Args:
        use_llm: Whether to use LLM (True) or template-based (False)
        model_name: Name of the LLM model if use_llm is True
    
    Returns:
        FoodExplainer or SimpleFoodExplainer instance
    """
    if use_llm:
        try:
            return FoodExplainer(model_name=model_name)
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("Falling back to simple template-based explainer")
            return SimpleFoodExplainer()
    else:
        return SimpleFoodExplainer()

