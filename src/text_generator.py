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
        
        # Track if we're using device_map for accelerate
        self.use_device_map = self.device == 'cuda'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.use_device_map else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        # Create pipeline
        # Don't pass device argument if using device_map (accelerate handles it)
        if self.use_device_map:
            # When using device_map, don't specify device (accelerate handles it)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
        else:
            # For CPU, explicitly set device
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1
            )
        
        print(f"Text generation model loaded successfully")
    
    def generate_explanation(
        self, 
        food_name: str, 
        food_info: Dict,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a natural language explanation about a Korean food
        
        Args:
            food_name: English name of the food
            food_info: Dictionary with food information (from knowledge base)
            max_new_tokens: Maximum number of new tokens to generate
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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                truncation=True
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract only the assistant's response
            # The format is: <|system|>...<|user|>...<|assistant|>RESPONSE</s>
            if '<|assistant|>' in generated_text:
                # Split at <|assistant|> and take everything after it
                parts = generated_text.split('<|assistant|>')
                if len(parts) > 1:
                    response = parts[-1].strip()
                    # Remove any trailing </s> or special tokens
                    response = response.split('</s>')[0].strip()
                    # Remove any text after newline followed by <| (start of new turn)
                    if '\n<|' in response:
                        response = response.split('\n<|')[0].strip()
                    return response
            
            # Fallback: try to extract after the prompt
            if len(generated_text) > len(prompt):
                response = generated_text[len(prompt):].strip()
                # Clean up special tokens
                response = response.split('</s>')[0].strip()
                response = response.split('<|')[0].strip()
                if response:
                    return response
            
            # If extraction fails, fall back to template
            print("Failed to extract response from LLM, using template")
            return self._generate_template_explanation(food_name, food_info)
        
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
        ingredients_str = ", ".join(ingredients[:5]) if ingredients else "traditional Korean ingredients"
        
        # Create prompt with context but let LLM generate the response
        prompt = f"""<|system|>
You are a Korean food expert. Provide clear and concise descriptions of Korean dishes in 1-2 short paragraphs.</s>
<|user|>
Tell me about {food_name} ({korean_name}). Here's some information about it:
- Category: {category}
- Description: {description}
- Main ingredients: {ingredients_str}
- Preparation: {cooking_method}
- Cultural significance: {cultural_note}

Explain this dish in a natural, conversational way.</s>
<|assistant|>
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

