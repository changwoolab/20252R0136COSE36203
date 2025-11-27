"""
Attribute Database for Korean Food Attributes
Stores descriptions for food attributes (flavors, cooking methods, ingredients, etc.)
"""
import json
import os
from typing import Dict, Optional, List


class AttributeDatabase:
    """Manages attribute descriptions for Korean food"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize the attribute database
        
        Args:
            db_path: Path to attribute database JSON file (optional)
        """
        self.db_path = db_path
        self.attributes = {}
        self._initialize_default_attributes()
        
        if db_path and os.path.exists(db_path):
            self.load(db_path)
    
    def _initialize_default_attributes(self):
        """Initialize with default Korean food attributes"""
        self.attributes = {
            # Flavor Profiles
            "Spicy": "A flavor profile characterized by Gochujang or red pepper powder (gochugaru), creating a fiery, pungent heat that is a signature of Korean cuisine.",
            "Mild": "A gentle, non-spicy flavor profile, often featuring subtle seasonings and natural flavors of ingredients.",
            "Sweet": "A flavor profile featuring sweetness from sugar, honey, or fruits like pear, creating a balanced taste.",
            "Savory": "A rich, umami flavor profile from fermented ingredients like soy sauce, doenjang, or fish sauce.",
            "Sour": "A tangy flavor profile from vinegar, fermented foods, or citrus elements.",
            "Salty": "A flavor profile from soy sauce, salt, or fermented seafood, providing depth and seasoning.",
            
            # Cooking Methods
            "Grilled": "Cooked over an open flame or on a grill, imparting a smoky, charred flavor and crispy texture.",
            "Fried": "Cooked in hot oil, creating a crispy, golden exterior while maintaining tender interior.",
            "Stewed": "Slow-cooked in liquid, allowing flavors to meld and ingredients to become tender and flavorful.",
            "Boiled": "Cooked in boiling liquid, often used for soups and noodles, creating a clear, clean flavor.",
            "Steamed": "Cooked with steam, preserving natural flavors and nutrients while keeping ingredients moist.",
            "Stir-fried": "Quickly cooked in a hot pan with oil, maintaining crisp textures and vibrant colors.",
            "Fermented": "Preserved through fermentation, developing complex, tangy flavors and beneficial probiotics.",
            "Raw": "Served fresh without cooking, highlighting natural textures and flavors.",
            
            # Main Ingredients
            "Chicken": "A common poultry ingredient in Korean cuisine, used in soups, stews, grilled dishes, and fried preparations.",
            "Beef": "A premium protein in Korean cooking, often marinated and grilled, or used in soups and stews.",
            "Pork": "A versatile meat used in Korean BBQ, stews, and various dishes, especially popular as pork belly.",
            "Seafood": "Fresh fish, shrimp, squid, and other marine ingredients, common in coastal Korean cuisine.",
            "Tofu": "Soft or firm soybean curd, a protein-rich ingredient used in stews and side dishes.",
            "Rice": "The staple grain of Korean cuisine, served with almost every meal, often sticky short-grain rice.",
            "Noodles": "Various types including wheat, buckwheat, and sweet potato noodles, used in soups and stir-fries.",
            "Vegetables": "Fresh vegetables like cabbage, spinach, bean sprouts, and radish, essential in Korean meals.",
            "Kimchi": "Fermented vegetables, most commonly napa cabbage, seasoned with chili and garlic, a Korean staple.",
            
            # Dish Types
            "Soup": "A liquid-based dish, often served as part of a meal, ranging from light broths to hearty stews.",
            "Stew": "A thick, hearty dish cooked slowly in liquid, with rich, concentrated flavors.",
            "Rice Dish": "A dish featuring rice as the main component, often mixed or topped with various ingredients.",
            "Noodle Dish": "A dish featuring noodles as the primary ingredient, served in soup or stir-fried.",
            "Side Dish": "Small dishes (banchan) served alongside rice and main dishes, providing variety and balance.",
            "Street Food": "Quick, portable foods sold at markets and street stalls, often spicy and flavorful.",
            "BBQ": "Grilled meat dishes, often cooked at the table, a social dining experience in Korean culture.",
            "Fried Food": "Crispy, deep-fried dishes, often served as snacks or with meals.",
            
            # Texture
            "Crispy": "A texture characterized by a crunchy, brittle exterior, often achieved through frying or grilling.",
            "Chewy": "A texture that requires some effort to bite through, like rice cakes or certain noodles.",
            "Tender": "A soft, easily chewable texture, often from slow cooking or marination.",
            "Soft": "A delicate, yielding texture, like soft tofu or well-cooked vegetables.",
            "Crunchy": "A firm, crisp texture that makes a sound when bitten, from fresh vegetables or fried coatings.",
        }
    
    def load(self, db_path: str):
        """Load attribute database from JSON file"""
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.attributes.update(loaded)
            print(f"Loaded attribute database with {len(self.attributes)} attributes")
        else:
            print(f"Attribute database not found at {db_path}, using default attributes")
    
    def save(self, db_path: str = None):
        """Save attribute database to JSON file"""
        path = db_path or self.db_path
        if not path:
            print("No path specified for saving attribute database")
            return
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.attributes, f, indent=2, ensure_ascii=False)
        print(f"Saved attribute database to {path}")
    
    def add_attribute(self, name: str, description: str):
        """Add or update an attribute"""
        self.attributes[name] = description
    
    def get_attribute(self, name: str) -> Optional[str]:
        """Get description for an attribute"""
        return self.attributes.get(name)
    
    def get_attributes(self, names: List[str]) -> Dict[str, str]:
        """Get descriptions for multiple attributes"""
        return {name: self.attributes.get(name, "") for name in names}
    
    def get_all_attributes(self) -> Dict[str, str]:
        """Get all attributes"""
        return self.attributes.copy()
    
    def get_attribute_list(self) -> List[str]:
        """Get list of all attribute names"""
        return list(self.attributes.keys())


def create_default_attribute_db(db_path: str):
    """Create a default attribute database file"""
    db = AttributeDatabase()
    db.save(db_path)
    return db

