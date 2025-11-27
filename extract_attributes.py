"""
Automated Attribute Extraction from Food Descriptions
Uses NLP and frequency analysis to extract attributes from existing GPT-5 descriptions
"""
import json
import os
import sys
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.knowledge_base import FoodKnowledgeBase


# Taxonomy categories for attributes
ATTRIBUTE_TAXONOMY = {
    "Visual (CLIP-detectable)": {
        "Color": ["red", "white", "clear", "brown", "golden", "black", "green", "yellow", "orange"],
        "Texture": ["crispy", "soupy", "grilled", "tender", "soft", "chewy", "crunchy", "smooth", "thick", "thin"],
        "Vessel": ["stone pot", "grill", "plate", "bowl", "pan", "pot", "skillet"]
    },
    "Ingredient (Contextual)": {
        "Main Protein": ["beef", "pork", "chicken", "seafood", "fish", "shrimp", "squid", "crab", "tofu", "egg"],
        "Carbs": ["rice", "noodle", "rice cake", "dumpling", "pancake", "bread"]
    },
    "Flavor (Inferred)": {
        "Flavor": ["spicy", "savory", "mild", "sweet", "sour", "salty", "bitter", "umami", "tangy", "pungent"]
    }
}

# Cooking methods (extracted from descriptions)
COOKING_METHODS = [
    "grilled", "fried", "stewed", "braised", "boiled", "steamed", 
    "stir-fried", "pan-fried", "deep-fried", "roasted", "marinated",
    "fermented", "raw", "sautéed", "simmered", "baked"
]

# Common descriptors to look for
COMMON_DESCRIPTORS = {
    "spicy": ["spicy", "hot", "fiery", "pungent", "gochujang", "gochugaru", "chili", "chili paste"],
    "mild": ["mild", "gentle", "subtle", "delicate"],
    "sweet": ["sweet", "sugar", "honey", "sweetened"],
    "savory": ["savory", "umami", "flavorful", "rich"],
    "sour": ["sour", "tangy", "vinegar", "fermented"],
    "salty": ["salty", "soy sauce", "salted"],
    "crispy": ["crispy", "crunchy", "crisp"],
    "tender": ["tender", "soft", "melting"],
    "chewy": ["chewy", "elastic", "firm"],
    "grilled": ["grilled", "charred", "smoky", "bbq"],
    "fried": ["fried", "deep-fried", "pan-fried"],
    "stewed": ["stewed", "braised", "simmered"],
    "soup": ["soup", "broth", "liquid-based"],
    "stew": ["stew", "jjigae", "thick"],
    "rice": ["rice", "rice dish", "rice-based"],
    "noodle": ["noodle", "noodles", "noodle dish"],
    "beef": ["beef", "beef short ribs", "beef belly"],
    "pork": ["pork", "pork belly", "pork ribs"],
    "chicken": ["chicken", "poultry"],
    "seafood": ["seafood", "fish", "shrimp", "squid", "crab", "mussel", "oyster"],
    "tofu": ["tofu", "bean curd"],
    "vegetables": ["vegetables", "vegetable", "cabbage", "spinach", "bean sprouts"],
    "kimchi": ["kimchi", "fermented vegetables"],
    "red": ["red", "reddish", "crimson"],
    "white": ["white", "pale"],
    "clear": ["clear", "transparent", "translucent"],
    "brown": ["brown", "browned", "golden brown"],
    "stone pot": ["stone pot", "dolsot", "earthenware"],
    "plate": ["plate", "served on plate"],
    "bowl": ["bowl", "served in bowl"]
}


def extract_attributes_from_text(text: str, food_name: str = "") -> Set[str]:
    """
    Extract attributes from a text description using keyword matching
    
    Args:
        text: Description text
        food_name: Food name (for context)
    
    Returns:
        Set of detected attribute names
    """
    text_lower = text.lower()
    attributes = set()
    
    # Check each descriptor category
    for attr_name, keywords in COMMON_DESCRIPTORS.items():
        for keyword in keywords:
            if keyword in text_lower:
                attributes.add(attr_name.capitalize())
                break
    
    # Extract cooking methods
    for method in COOKING_METHODS:
        if method in text_lower:
            attributes.add(method.capitalize())
    
    # Extract from food name
    food_lower = food_name.lower()
    if "soup" in food_lower or "guk" in food_lower or "tang" in food_lower:
        attributes.add("Soup")
    if "stew" in food_lower or "jjigae" in food_lower or "jjim" in food_lower:
        attributes.add("Stew")
    if "fried" in food_lower or "twigim" in food_lower:
        attributes.add("Fried")
    if "grilled" in food_lower or "gui" in food_lower:
        attributes.add("Grilled")
    if "rice" in food_lower or "bap" in food_lower:
        attributes.add("Rice")
    if "noodle" in food_lower or "myeon" in food_lower:
        attributes.add("Noodle")
    if "kimchi" in food_lower:
        attributes.add("Kimchi")
    
    return attributes


def analyze_all_descriptions(kb_path: str) -> Dict[str, Dict]:
    """
    Analyze all food descriptions to extract attributes using frequency analysis
    
    Args:
        kb_path: Path to knowledge base JSON
    
    Returns:
        Dictionary with attribute frequencies and mappings
    """
    kb = FoodKnowledgeBase(kb_path)
    all_foods = kb.get_all_foods()
    
    # Collect all attributes
    attribute_counts = Counter()
    attribute_to_foods = defaultdict(list)
    food_to_attributes = {}
    
    print(f"Analyzing {len(all_foods)} food descriptions...")
    
    for food_name, food_info in all_foods.items():
        # Combine all text fields
        description = food_info.get('description', '')
        cooking_method = food_info.get('cooking_method', '')
        category = food_info.get('category', '')
        ingredients = ' '.join(food_info.get('ingredients', []))
        
        combined_text = f"{description} {cooking_method} {category} {ingredients}".lower()
        
        # Extract attributes
        attributes = extract_attributes_from_text(combined_text, food_name)
        
        # Also check existing attributes field
        existing_attrs = food_info.get('attributes', '')
        if existing_attrs:
            # Parse existing attributes (comma-separated)
            for attr in existing_attrs.split(','):
                attr_clean = attr.strip().lower()
                if attr_clean:
                    attributes.add(attr_clean.capitalize())
        
        food_to_attributes[food_name] = attributes
        
        # Count attributes
        for attr in attributes:
            attribute_counts[attr] += 1
            attribute_to_foods[attr].append(food_name)
    
    return {
        'attribute_counts': attribute_counts,
        'attribute_to_foods': dict(attribute_to_foods),
        'food_to_attributes': food_to_attributes
    }


def generate_attribute_description(attribute_name: str, example_foods: List[str], kb: FoodKnowledgeBase) -> str:
    """
    Generate a generic description fragment for an attribute based on examples
    
    Args:
        attribute_name: Name of the attribute
        example_foods: List of food names that have this attribute
        kb: Knowledge base instance
    
    Returns:
        Generic description fragment
    """
    # Get descriptions from example foods
    descriptions = []
    for food_name in example_foods[:5]:  # Use top 5 examples
        food_info = kb.get_food_info(food_name)
        if food_info:
            desc = food_info.get('description', '')
            if desc:
                descriptions.append(desc)
    
    # Generate description based on attribute type
    attr_lower = attribute_name.lower()
    
    # Flavor attributes
    if attr_lower in ["spicy", "hot"]:
        return "A flavor profile characterized by Gochujang (red chili paste) or red pepper powder (gochugaru), creating a fiery, pungent heat that is a signature of Korean cuisine."
    elif attr_lower == "mild":
        return "A gentle, non-spicy flavor profile, often featuring subtle seasonings and natural flavors of ingredients."
    elif attr_lower == "sweet":
        return "A flavor profile featuring sweetness from sugar, honey, or fruits like pear, creating a balanced taste."
    elif attr_lower in ["savory", "umami"]:
        return "A rich, umami flavor profile from fermented ingredients like soy sauce, doenjang, or fish sauce."
    elif attr_lower in ["sour", "tangy"]:
        return "A tangy flavor profile from vinegar, fermented foods, or citrus elements."
    elif attr_lower == "salty":
        return "A flavor profile from soy sauce, salt, or fermented seafood, providing depth and seasoning."
    
    # Cooking methods
    elif attr_lower == "grilled":
        return "Cooked over an open flame or on a grill, imparting a smoky, charred flavor and crispy texture."
    elif attr_lower == "fried":
        return "Cooked in hot oil, creating a crispy, golden exterior while maintaining tender interior."
    elif attr_lower in ["stewed", "braised"]:
        return "Slow-cooked in liquid, allowing flavors to meld and ingredients to become tender and flavorful."
    elif attr_lower == "boiled":
        return "Cooked in boiling liquid, often used for soups and noodles, creating a clear, clean flavor."
    elif attr_lower == "steamed":
        return "Cooked with steam, preserving natural flavors and nutrients while keeping ingredients moist."
    elif attr_lower == "stir-fried":
        return "Quickly cooked in a hot pan with oil, maintaining crisp textures and vibrant colors."
    elif attr_lower == "fermented":
        return "Preserved through fermentation, developing complex, tangy flavors and beneficial probiotics."
    
    # Ingredients
    elif attr_lower == "beef":
        return "A premium protein in Korean cooking, often marinated and grilled, or used in soups and stews."
    elif attr_lower == "pork":
        return "A versatile meat used in Korean BBQ, stews, and various dishes, especially popular as pork belly."
    elif attr_lower == "chicken":
        return "A common poultry ingredient in Korean cuisine, used in soups, stews, grilled dishes, and fried preparations."
    elif attr_lower == "seafood":
        return "Fresh fish, shrimp, squid, and other marine ingredients, common in coastal Korean cuisine."
    elif attr_lower == "tofu":
        return "Soft or firm soybean curd, a protein-rich ingredient used in stews and side dishes."
    elif attr_lower == "rice":
        return "The staple grain of Korean cuisine, served with almost every meal, often sticky short-grain rice."
    elif attr_lower == "noodle":
        return "Various types including wheat, buckwheat, and sweet potato noodles, used in soups and stir-fries."
    elif attr_lower == "vegetables":
        return "Fresh vegetables like cabbage, spinach, bean sprouts, and radish, essential in Korean meals."
    elif attr_lower == "kimchi":
        return "Fermented vegetables, most commonly napa cabbage, seasoned with chili and garlic, a Korean staple."
    
    # Dish types
    elif attr_lower == "soup":
        return "A liquid-based dish, often served as part of a meal, ranging from light broths to hearty stews."
    elif attr_lower == "stew":
        return "A thick, hearty dish cooked slowly in liquid, with rich, concentrated flavors."
    
    # Textures
    elif attr_lower == "crispy":
        return "A texture characterized by a crunchy, brittle exterior, often achieved through frying or grilling."
    elif attr_lower == "tender":
        return "A soft, easily chewable texture, often from slow cooking or marination."
    elif attr_lower == "chewy":
        return "A texture that requires some effort to bite through, like rice cakes or certain noodles."
    elif attr_lower == "soft":
        return "A delicate, yielding texture, like soft tofu or well-cooked vegetables."
    elif attr_lower == "crunchy":
        return "A firm, crisp texture that makes a sound when bitten, from fresh vegetables or fried coatings."
    
    # Colors
    elif attr_lower == "red":
        return "A reddish color, often from gochujang, gochugaru, or other red ingredients."
    elif attr_lower == "white":
        return "A pale or white color, often from rice, tofu, or clear broths."
    elif attr_lower == "clear":
        return "A transparent or translucent appearance, often from clear broths or soups."
    elif attr_lower == "brown":
        return "A brown or golden brown color, often from soy sauce, caramelization, or grilling."
    
    # Vessels
    elif attr_lower == "stone pot":
        return "Served in a stone pot (dolsot) or earthenware, keeping the dish hot and adding rustic flavor."
    elif attr_lower == "plate":
        return "Served on a plate, typically for dry dishes or grilled items."
    elif attr_lower == "bowl":
        return "Served in a bowl, typically for soups, stews, or rice dishes."
    
    # Default: generate from examples
    if descriptions:
        # Extract common patterns
        words = []
        for desc in descriptions:
            words.extend(desc.lower().split())
        
        # Find common descriptive words
        word_counts = Counter(words)
        common_words = [w for w, c in word_counts.most_common(10) if len(w) > 3 and w not in ['the', 'and', 'with', 'from', 'that', 'this', 'dish', 'korean']]
        
        if common_words:
            return f"A characteristic of Korean cuisine related to {attribute_name.lower()}, often described as {', '.join(common_words[:3])}."
    
    # Final fallback
    return f"A characteristic of Korean cuisine: {attribute_name.lower()}."


def build_attribute_database(kb_path: str, output_path: str, min_frequency: int = 2):
    """
    Build attribute database from food descriptions
    
    Args:
        kb_path: Path to knowledge base JSON
        output_path: Path to save attribute database JSON
        min_frequency: Minimum frequency for an attribute to be included
    """
    print("=" * 70)
    print("Automated Attribute Extraction from Food Descriptions")
    print("=" * 70)
    
    # Analyze descriptions
    analysis = analyze_all_descriptions(kb_path)
    kb = FoodKnowledgeBase(kb_path)
    
    attribute_counts = analysis['attribute_counts']
    attribute_to_foods = analysis['attribute_to_foods']
    
    print(f"\nFound {len(attribute_counts)} unique attributes")
    print(f"Filtering attributes with frequency >= {min_frequency}...")
    
    # Filter by frequency
    filtered_attributes = {
        attr: count for attr, count in attribute_counts.items() 
        if count >= min_frequency
    }
    
    print(f"Keeping {len(filtered_attributes)} attributes after filtering")
    
    # Generate attribute database
    attribute_db = {}
    
    print("\nGenerating attribute descriptions...")
    for attr_name, count in sorted(filtered_attributes.items(), key=lambda x: x[1], reverse=True):
        example_foods = attribute_to_foods[attr_name]
        description = generate_attribute_description(attr_name, example_foods, kb)
        attribute_db[attr_name] = description
        
        print(f"  {attr_name}: {count} foods, {len(description)} chars")
    
    # Save attribute database
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(attribute_db, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Attribute database saved to: {output_path}")
    print(f"  Total attributes: {len(attribute_db)}")
    
    # Show statistics by category
    print("\nAttribute Statistics by Category:")
    visual_count = sum(1 for attr in attribute_db.keys() if any(cat in attr.lower() for cat in ['crispy', 'tender', 'chewy', 'soft', 'red', 'white', 'clear', 'brown', 'stone', 'plate', 'bowl']))
    flavor_count = sum(1 for attr in attribute_db.keys() if any(cat in attr.lower() for cat in ['spicy', 'mild', 'sweet', 'savory', 'sour', 'salty']))
    ingredient_count = sum(1 for attr in attribute_db.keys() if any(cat in attr.lower() for cat in ['beef', 'pork', 'chicken', 'seafood', 'tofu', 'rice', 'noodle', 'vegetable', 'kimchi']))
    method_count = sum(1 for attr in attribute_db.keys() if any(cat in attr.lower() for cat in ['grilled', 'fried', 'stewed', 'braised', 'boiled', 'steamed', 'stir-fried', 'fermented']))
    
    print(f"  Visual (Texture/Color/Vessel): {visual_count}")
    print(f"  Flavor: {flavor_count}")
    print(f"  Ingredient: {ingredient_count}")
    print(f"  Cooking Method: {method_count}")
    
    print("\n" + "=" * 70)
    print("Attribute extraction complete!")
    print("=" * 70)
    
    return attribute_db


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract attributes from food descriptions')
    parser.add_argument('--kb-path', type=str, default=config.DB_PATH,
                        help='Path to knowledge base JSON')
    parser.add_argument('--output', type=str, default=config.ATTRIBUTE_DB_PATH,
                        help='Output path for attribute database')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum frequency for an attribute to be included')
    
    args = parser.parse_args()
    
    build_attribute_database(args.kb_path, args.output, args.min_frequency)


if __name__ == "__main__":
    main()

