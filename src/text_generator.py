"""
Text Generator for Korean Food Explanations
Uses TinyLlama or similar small LLM to generate natural language explanations
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Optional, List, Tuple
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
    
    def generate_explanation_from_attributes(
        self,
        food_name: str,
        attribute_descriptions: Dict[str, str],
        predicted_attributes: List[Tuple[str, float]],
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate explanation from attributes when food is not in main DB
        Implements Strategy 2: Attribute Retrieval
        
        Args:
            food_name: Predicted food name (e.g., "Dakbal")
            attribute_descriptions: Dictionary mapping attribute names to descriptions
            predicted_attributes: List of (attribute_name, confidence) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated explanation text
        """
        # Create prompt with attributes
        attribute_names = [attr[0] for attr in predicted_attributes]
        attribute_texts = [attribute_descriptions.get(name, "") for name in attribute_names if name in attribute_descriptions]
        attribute_text = " ".join(attribute_texts)
        
        prompt = f"""<|system|>
You are a Korean food expert. Combine known attributes to describe a dish that may not be in your database.</s>
<|user|>
User image depicts: {food_name}. Known attributes: {', '.join(attribute_names)}. 
Attribute descriptions: {attribute_text}

Combine these facts into a description of {food_name}.</s>
<|assistant|>
"""
        
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
            
            # Extract response
            if '<|assistant|>' in generated_text:
                parts = generated_text.split('<|assistant|>')
                if len(parts) > 1:
                    response = parts[-1].strip()
                    response = response.split('</s>')[0].strip()
                    if '\n<|' in response:
                        response = response.split('\n<|')[0].strip()
                    if response:
                        return response
            
            # Fallback
            if len(generated_text) > len(prompt):
                response = generated_text[len(prompt):].strip()
                response = response.split('</s>')[0].strip()
                response = response.split('<|')[0].strip()
                if response:
                    return response
            
            # Template fallback
            return f"This is {food_name}. It is a dish with the following characteristics: {attribute_text}"
        
        except Exception as e:
            print(f"Error generating text from attributes: {e}")
            # Template fallback
            return f"This is {food_name}. It is a dish with the following characteristics: {attribute_text}"
    
    def generate_explanation(
        self, 
        food_name: str, 
        food_info: Dict,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        predicted_attributes: List[str] = None
    ) -> str:
        """
        Generate a natural language explanation about a Korean food (in-distribution)
        
        Args:
            food_name: English name of the food
            food_info: Dictionary with food information (from knowledge base)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            predicted_attributes: Optional list of predicted visual attributes
        
        Returns:
            Generated explanation text
        """
        # Create prompt with food information and attributes (in-distribution)
        prompt = self._create_prompt(food_name, food_info, predicted_attributes)
        
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
    
    def _create_prompt(self, food_name: str, food_info: Dict, predicted_attributes: List[str] = None) -> str:
        """Create a prompt for the LLM (in-distribution foods with full KB info + attributes)"""
        
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        category = food_info.get('category', '')
        ingredients = food_info.get('ingredients', [])
        cooking_method = food_info.get('cooking_method', '')
        cultural_note = food_info.get('cultural_note', '')
        
        # Format ingredients
        ingredients_str = ", ".join(ingredients[:5]) if ingredients else "traditional Korean ingredients"
        
        # Format attributes (in-distribution foods include visual attributes)
        attributes_str = ""
        if predicted_attributes:
            attributes_str = f"\n- Visual attributes: {', '.join(predicted_attributes[:5])}"
        
        # Create prompt with full KB context + attributes for in-distribution foods
        prompt = f"""<|system|>
You are a Korean food expert. Provide clear and concise descriptions of Korean dishes in 1-2 short paragraphs.</s>
<|user|>
Tell me about {food_name} ({korean_name}). Here's detailed information about it:
- Category: {category}
- Description: {description}
- Main ingredients: {ingredients_str}
- Preparation: {cooking_method}
- Cultural significance: {cultural_note}{attributes_str}

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
    
    def generate_zeroshot_explanation(
        self,
        food_name: str,
        food_info: Dict,
        max_new_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Generate explanation for zero-shot predictions using similar KB foods as context
        
        Args:
            food_name: Predicted food name (not in KB)
            food_info: Dictionary with food info including similar_kb_foods and similar_context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated explanation text
        """
        # Get similar KB foods context
        similar_kb_foods = food_info.get('similar_kb_foods', [])
        similar_kb_foods = similar_kb_foods[:1]
        print("Similar KB foods: ", similar_kb_foods)
        
        # Build simple prompt with only food name and similar dishes (zero-shot - no detailed attributes)
        similar_foods_str = ", ".join([f['name'] for f in similar_kb_foods]) if similar_kb_foods else ""
        
        # Simple prompt for zero-shot: only food name + similar dishes
        if similar_foods_str:
            prompt = f"""<|system|>
You are a Korean food expert. Provide helpful short descriptions of Korean dishes.</s>
<|user|>
Tell me about the Korean dish "{food_name}". 
It appears similar to these Korean dishes: {similar_foods_str}.

Write a brief 1-2 sentence description of {food_name}.</s>
<|assistant|>
"""
        else:
            # No similar foods found - just describe based on name
            prompt = f"""<|system|>
You are a Korean food expert. Provide helpful descriptions of Korean dishes.</s>
<|user|>
Tell me about the Korean dish "{food_name}".

Write a brief 2-3 sentence description of {food_name}.</s>
<|assistant|>
"""
        
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
            
            # Extract response
            if '<|assistant|>' in generated_text:
                parts = generated_text.split('<|assistant|>')
                if len(parts) > 1:
                    response = parts[-1].strip()
                    response = response.split('</s>')[0].strip()
                    if '\n<|' in response:
                        response = response.split('\n<|')[0].strip()
                    if response:
                        return response
            
            # Fallback
            if len(generated_text) > len(prompt):
                response = generated_text[len(prompt):].strip()
                response = response.split('</s>')[0].strip()
                response = response.split('<|')[0].strip()
                if response:
                    return response
            
            # Template fallback for zero-shot
            return self._generate_zeroshot_template(food_name, similar_kb_foods, inferred_category, similar_ingredients)
        
        except Exception as e:
            print(f"Error generating zero-shot explanation: {e}")
            return self._generate_zeroshot_template(food_name, similar_kb_foods, inferred_category, similar_ingredients)
    
    def _generate_zeroshot_template(
        self,
        food_name: str,
        similar_kb_foods: List[Dict],
        category: str = None,
        ingredients: List[str] = None
    ) -> str:
        """Template-based fallback for zero-shot explanations (simple - only food name + similar dishes)"""
        similar_names = [f['name'] for f in similar_kb_foods] if similar_kb_foods else []
        
        explanation = f"{food_name} is a traditional Korean dish. "
        
        if similar_names:
            explanation += f"It shares similarities with {', '.join(similar_names[:2])}."
        
        return explanation

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
    
    def generate_explanation_from_attributes(
        self,
        food_name: str,
        attribute_descriptions: Dict[str, str],
        predicted_attributes: List[Tuple[str, float]]
    ) -> str:
        """Generate explanation from attributes (template-based)"""
        attribute_names = [attr[0] for attr in predicted_attributes]
        attribute_texts = [attribute_descriptions.get(name, "") for name in attribute_names if name in attribute_descriptions]
        
        explanation = f"**{food_name}**\n\n"
        explanation += f"This dish is characterized by the following attributes:\n\n"
        
        for name, desc in attribute_descriptions.items():
            if desc:
                explanation += f"- **{name}**: {desc}\n\n"
        
        return explanation.strip()
    
    def generate_short_summary(self, food_name: str, food_info: Dict) -> str:
        """Generate a short summary"""
        korean_name = food_info.get('korean_name', '')
        description = food_info.get('description', '')
        first_sentence = description.split('.')[0] + '.'
        return f"{food_name} ({korean_name}): {first_sentence}"


def create_explainer(
    use_llm: bool = False,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
) -> object:
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

