"""
Build Knowledge Base for Korean Food
Generates descriptions for all Korean foods in the dataset
"""
import pandas as pd
import os
import sys
from src.knowledge_base import FoodKnowledgeBase, KOREAN_FOOD_DESCRIPTIONS
import config


# Extended descriptions for all Korean foods
def generate_food_descriptions():
    """Generate comprehensive descriptions for all Korean foods"""
    
    descriptions = {}
    
    # Read the Korean-English mapping
    df = pd.read_csv(config.CSV_PATH)
    
    for idx, row in df.iterrows():
        korean_name = row['korean name']
        english_name = row['english name']
        
        # Check if we have a predefined description
        if english_name in KOREAN_FOOD_DESCRIPTIONS:
            info = KOREAN_FOOD_DESCRIPTIONS[english_name]
            descriptions[english_name] = {
                'korean_name': korean_name,
                'english_name': english_name,
                'category': info['category'],
                'description': info['description'],
                'ingredients': info['ingredients'],
                'cooking_method': info['cooking_method'],
                'cultural_note': info['cultural_note']
            }
        else:
            # Generate a basic description for foods not in the predefined list
            category = classify_food_category(english_name)
            descriptions[english_name] = {
                'korean_name': korean_name,
                'english_name': english_name,
                'category': category,
                'description': generate_basic_description(english_name, category),
                'ingredients': extract_likely_ingredients(english_name),
                'cooking_method': infer_cooking_method(english_name),
                'cultural_note': "A traditional Korean dish enjoyed across the country."
            }
    
    return descriptions


def classify_food_category(food_name):
    """Classify food into categories based on name"""
    name_lower = food_name.lower()
    
    if any(word in name_lower for word in ['stew', 'soup', 'jjigae', 'tang', 'guk']):
        return "Soup/Stew"
    elif any(word in name_lower for word in ['grilled', 'bbq', 'gui']):
        return "Grilled Meat"
    elif any(word in name_lower for word in ['kimchi']):
        return "Kimchi"
    elif any(word in name_lower for word in ['noodle', 'myeon']):
        return "Noodles"
    elif any(word in name_lower for word in ['rice', 'bap', 'porridge', 'juk']):
        return "Rice Dish"
    elif any(word in name_lower for word in ['pancake', 'jeon']):
        return "Pancake"
    elif any(word in name_lower for word in ['fried', 'tuigim']):
        return "Fried Food"
    elif any(word in name_lower for word in ['seasoned', 'muchim', 'namul']):
        return "Side Dish (Banchan)"
    elif any(word in name_lower for word in ['braised', 'jorim']):
        return "Braised Dish"
    elif any(word in name_lower for word in ['cake', 'tteok', 'sweet', 'dessert', 'punch', 'drink']):
        return "Dessert/Beverage"
    elif any(word in name_lower for word in ['wraps', 'ssam', 'slices']):
        return "Meat Dish"
    else:
        return "Korean Dish"


def generate_basic_description(food_name, category):
    """Generate a basic description based on food name and category"""
    
    name_lower = food_name.lower()
    
    # Extract main ingredient from name
    main_ingredient = ""
    if 'beef' in name_lower:
        main_ingredient = "beef"
    elif 'pork' in name_lower:
        main_ingredient = "pork"
    elif 'chicken' in name_lower:
        main_ingredient = "chicken"
    elif 'fish' in name_lower or 'seafood' in name_lower:
        main_ingredient = "seafood"
    elif 'tofu' in name_lower:
        main_ingredient = "tofu"
    elif 'vegetable' in name_lower or 'radish' in name_lower or 'potato' in name_lower:
        main_ingredient = "vegetables"
    
    # Generate description based on category and ingredients
    if category == "Soup/Stew":
        return f"A traditional Korean soup/stew featuring {main_ingredient if main_ingredient else 'flavorful ingredients'}, simmered in a savory broth with vegetables and seasonings."
    elif category == "Grilled Meat":
        return f"Korean-style grilled {main_ingredient if main_ingredient else 'meat'}, often marinated in a flavorful sauce and cooked to perfection."
    elif category == "Kimchi":
        return f"A fermented vegetable dish, seasoned with chili powder, garlic, and other spices. An essential part of Korean cuisine."
    elif category == "Noodles":
        return f"A Korean noodle dish featuring {main_ingredient if main_ingredient else 'savory toppings'} in a flavorful broth or sauce."
    elif category == "Rice Dish":
        return f"A Korean rice-based dish with {main_ingredient if main_ingredient else 'various toppings'}, providing a satisfying and complete meal."
    elif category == "Pancake":
        return f"A savory Korean pancake made with {main_ingredient if main_ingredient else 'vegetables'}, pan-fried until crispy on the outside."
    elif category == "Side Dish (Banchan)":
        return f"A traditional Korean side dish featuring {main_ingredient if main_ingredient else 'vegetables'}, seasoned with sesame oil and other seasonings."
    elif category == "Braised Dish":
        return f"A braised dish where {main_ingredient if main_ingredient else 'ingredients'} are slow-cooked in a savory sauce until tender and flavorful."
    elif category == "Fried Food":
        return f"Crispy fried {main_ingredient if main_ingredient else 'dish'} with Korean-style seasoning and preparation."
    else:
        return f"A traditional Korean dish made with {main_ingredient if main_ingredient else 'quality ingredients'} and authentic Korean seasonings."


def extract_likely_ingredients(food_name):
    """Extract likely ingredients from food name"""
    ingredients = []
    name_lower = food_name.lower()
    
    # Common proteins
    if 'beef' in name_lower:
        ingredients.append("beef")
    if 'pork' in name_lower:
        ingredients.append("pork")
    if 'chicken' in name_lower:
        ingredients.append("chicken")
    if 'fish' in name_lower or 'seafood' in name_lower:
        ingredients.append("fish or seafood")
    if 'tofu' in name_lower:
        ingredients.append("tofu")
    if 'egg' in name_lower:
        ingredients.append("egg")
    
    # Common vegetables
    if 'kimchi' in name_lower:
        ingredients.extend(["kimchi", "gochugaru (chili powder)"])
    if 'radish' in name_lower:
        ingredients.append("radish")
    if 'potato' in name_lower:
        ingredients.append("potato")
    if 'mushroom' in name_lower:
        ingredients.append("mushrooms")
    if 'spinach' in name_lower:
        ingredients.append("spinach")
    if 'bean' in name_lower:
        ingredients.append("beans")
    if 'noodle' in name_lower:
        ingredients.append("noodles")
    if 'rice' in name_lower:
        ingredients.append("rice")
    
    # Common seasonings
    ingredients.extend(["garlic", "sesame oil", "soy sauce"])
    
    return ingredients if ingredients else ["traditional Korean ingredients"]


def infer_cooking_method(food_name):
    """Infer cooking method from food name"""
    name_lower = food_name.lower()
    
    if 'grilled' in name_lower:
        return "Grilled over high heat"
    elif 'fried' in name_lower:
        return "Deep-fried until crispy"
    elif 'stew' in name_lower or 'soup' in name_lower:
        return "Simmered in broth"
    elif 'braised' in name_lower:
        return "Braised in sauce until tender"
    elif 'steamed' in name_lower:
        return "Steamed to perfection"
    elif 'stir-fried' in name_lower:
        return "Stir-fried in a wok or pan"
    elif 'pancake' in name_lower:
        return "Pan-fried until golden"
    elif 'marinated' in name_lower:
        return "Marinated and then cooked"
    elif 'seasoned' in name_lower:
        return "Seasoned with Korean spices and sauces"
    else:
        return "Prepared using traditional Korean cooking methods"


def main():
    print("Building Korean Food Knowledge Base...")
    print(f"Reading data from: {config.CSV_PATH}")
    
    # Generate descriptions
    descriptions = generate_food_descriptions()
    
    print(f"\nGenerated descriptions for {len(descriptions)} foods")
    
    # Create knowledge base
    kb = FoodKnowledgeBase(config.DB_PATH)
    
    # Add all foods to knowledge base
    for food_name, info in descriptions.items():
        kb.add_food(
            english_name=info['english_name'],
            korean_name=info['korean_name'],
            description=info['description'],
            category=info['category'],
            ingredients=info['ingredients'],
            cooking_method=info['cooking_method'],
            cultural_note=info['cultural_note']
        )
    
    # Save knowledge base
    kb.save()
    
    print(f"\n✓ Knowledge base successfully created at {config.DB_PATH}")
    print(f"✓ Total foods in database: {len(kb.get_food_names())}")
    
    # Show some examples
    print("\n--- Sample Entries ---")
    for food_name in list(kb.get_food_names())[:5]:
        info = kb.get_food_info(food_name)
        print(f"\n{info['english_name']} ({info['korean_name']})")
        print(f"Category: {info['category']}")
        print(f"Description: {info['description'][:100]}...")


if __name__ == "__main__":
    main()

