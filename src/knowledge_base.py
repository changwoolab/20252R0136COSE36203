"""
Knowledge Base for Korean Food Descriptions
"""
import json
import os
from typing import Dict, Optional


class FoodKnowledgeBase:
    """Manages Korean food descriptions and information"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.knowledge = {}
        self.load()
    
    def load(self):
        """Load knowledge base from JSON file"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            print(f"Loaded knowledge base with {len(self.knowledge)} entries")
        else:
            print("No existing knowledge base found")
    
    def save(self):
        """Save knowledge base to JSON file"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        print(f"Saved knowledge base to {self.db_path}")
    
    def add_food(self, english_name: str, korean_name: str, description: str, 
                 category: str = "", ingredients: list = None, 
                 cooking_method: str = "", cultural_note: str = ""):
        """Add or update food information"""
        self.knowledge[english_name] = {
            "korean_name": korean_name,
            "english_name": english_name,
            "description": description,
            "category": category,
            "ingredients": ingredients or [],
            "cooking_method": cooking_method,
            "cultural_note": cultural_note
        }
    
    def get_food_info(self, english_name: str) -> Optional[Dict]:
        """Retrieve food information by English name"""
        return self.knowledge.get(english_name)
    
    def get_all_foods(self) -> Dict:
        """Get all food entries"""
        return self.knowledge
    
    def get_food_names(self) -> list:
        """Get list of all English food names"""
        return list(self.knowledge.keys())


# Comprehensive food descriptions database
KOREAN_FOOD_DESCRIPTIONS = {
    "Bibimbap": {
        "category": "Rice Dish",
        "description": "A vibrant mixed rice dish topped with seasoned vegetables, beef, a fried egg, and gochujang (Korean chili paste). The name literally means 'mixed rice'.",
        "ingredients": ["rice", "vegetables (spinach, bean sprouts, carrots, mushrooms)", "beef", "egg", "gochujang", "sesame oil"],
        "cooking_method": "Each ingredient is prepared separately and arranged over warm rice. Mixed together before eating.",
        "cultural_note": "One of the most iconic Korean dishes, representing harmony and balance with its colorful ingredients."
    },
    "Gimbap": {
        "category": "Rice Dish",
        "description": "Korean-style seaweed rice rolls filled with vegetables, egg, and often meat or fish. Similar in appearance to sushi but with distinctly Korean flavors.",
        "ingredients": ["rice", "seaweed", "vegetables", "egg", "pickled radish", "meat or fish"],
        "cooking_method": "Seasoned rice and fillings are rolled in dried seaweed sheets and sliced into bite-sized pieces.",
        "cultural_note": "A popular picnic and lunchbox food, beloved by Koreans of all ages."
    },
    "Kimchi Stew": {
        "category": "Stew",
        "description": "A spicy, hearty stew made with aged kimchi, pork, tofu, and vegetables. The longer-fermented kimchi provides deeper, more complex flavors.",
        "ingredients": ["aged kimchi", "pork belly", "tofu", "onions", "green onions", "gochugaru"],
        "cooking_method": "Kimchi is sautéed with pork, then simmered in broth with tofu and other ingredients.",
        "cultural_note": "A comfort food staple in Korean households, especially popular in cold weather."
    },
    "Bulgogi": {
        "category": "Grilled Meat",
        "description": "Thinly sliced beef marinated in a sweet and savory sauce made with soy sauce, sugar, garlic, and sesame oil, then grilled or stir-fried.",
        "ingredients": ["beef", "soy sauce", "sugar", "garlic", "sesame oil", "pear", "onions"],
        "cooking_method": "Meat is marinated for several hours, then grilled over high heat or stir-fried.",
        "cultural_note": "Literally means 'fire meat'. One of the most popular Korean dishes internationally."
    },
    "Marinated Grilled Beef": {
        "category": "Grilled Meat",
        "description": "Thinly sliced beef marinated in a sweet and savory sauce made with soy sauce, sugar, garlic, and sesame oil, then grilled or stir-fried.",
        "ingredients": ["beef", "soy sauce", "sugar", "garlic", "sesame oil", "pear", "onions"],
        "cooking_method": "Meat is marinated for several hours, then grilled over high heat or stir-fried.",
        "cultural_note": "Literally means 'fire meat'. One of the most popular Korean dishes internationally."
    },
    "Grilled Pork Belly": {
        "category": "Grilled Meat",
        "description": "Thick slices of pork belly grilled at the table and typically wrapped in lettuce with garlic, peppers, and ssamjang (Korean dipping sauce).",
        "ingredients": ["pork belly", "lettuce", "garlic", "peppers", "ssamjang", "kimchi"],
        "cooking_method": "Pork is grilled on a tabletop grill and eaten with various accompaniments.",
        "cultural_note": "A favorite at Korean BBQ restaurants, often enjoyed with soju."
    },
    "Samgyeopsal": {
        "category": "Grilled Meat",
        "description": "Thick slices of pork belly grilled at the table and typically wrapped in lettuce with garlic, peppers, and ssamjang.",
        "ingredients": ["pork belly", "lettuce", "garlic", "peppers", "ssamjang", "kimchi"],
        "cooking_method": "Pork is grilled on a tabletop grill and eaten with various accompaniments.",
        "cultural_note": "A favorite at Korean BBQ restaurants, often enjoyed with soju."
    },
    "Ginseng Chicken Soup": {
        "category": "Soup",
        "description": "A nourishing soup made with a whole young chicken stuffed with glutinous rice, ginseng, jujube, and garlic, simmered until tender.",
        "ingredients": ["whole young chicken", "glutinous rice", "Korean ginseng", "jujube", "garlic", "ginger"],
        "cooking_method": "Chicken is stuffed with rice and ingredients, then slow-cooked in broth.",
        "cultural_note": "Traditionally eaten during the hottest days of summer (boknal) to restore energy."
    },
    "Samgyetang": {
        "category": "Soup",
        "description": "A nourishing soup made with a whole young chicken stuffed with glutinous rice, ginseng, jujube, and garlic, simmered until tender.",
        "ingredients": ["whole young chicken", "glutinous rice", "Korean ginseng", "jujube", "garlic", "ginger"],
        "cooking_method": "Chicken is stuffed with rice and ingredients, then slow-cooked in broth.",
        "cultural_note": "Traditionally eaten during the hottest days of summer (boknal) to restore energy."
    },
    "Spicy Rice Cakes": {
        "category": "Street Food",
        "description": "Chewy cylindrical rice cakes cooked in a sweet and spicy red chili sauce, often with fish cakes, boiled eggs, and scallions.",
        "ingredients": ["rice cakes", "gochujang", "gochugaru", "sugar", "fish cakes", "scallions"],
        "cooking_method": "Rice cakes are simmered in a spicy-sweet sauce until tender and well-coated.",
        "cultural_note": "One of the most popular Korean street foods, beloved by students and office workers."
    },
    "Tteokbokki": {
        "category": "Street Food",
        "description": "Chewy cylindrical rice cakes cooked in a sweet and spicy red chili sauce, often with fish cakes, boiled eggs, and scallions.",
        "ingredients": ["rice cakes", "gochujang", "gochugaru", "sugar", "fish cakes", "scallions"],
        "cooking_method": "Rice cakes are simmered in a spicy-sweet sauce until tender and well-coated.",
        "cultural_note": "One of the most popular Korean street foods, beloved by students and office workers."
    },
    "Cold Buckwheat Noodles": {
        "category": "Noodles",
        "description": "Thin buckwheat noodles served in a cold, refreshing broth, topped with sliced pear, cucumber, radish, egg, and sometimes beef.",
        "ingredients": ["buckwheat noodles", "cold beef broth", "cucumber", "pear", "egg", "radish"],
        "cooking_method": "Noodles are cooked and chilled, then served in ice-cold broth.",
        "cultural_note": "A summer favorite, especially popular in North Korean cuisine."
    },
    "Mul Naengmyeon": {
        "category": "Noodles",
        "description": "Thin buckwheat noodles served in a cold, refreshing broth, topped with sliced pear, cucumber, radish, egg, and sometimes beef.",
        "ingredients": ["buckwheat noodles", "cold beef broth", "cucumber", "pear", "egg", "radish"],
        "cooking_method": "Noodles are cooked and chilled, then served in ice-cold broth.",
        "cultural_note": "A summer favorite, especially popular in North Korean cuisine."
    },
    "Soybean Paste Stew": {
        "category": "Stew",
        "description": "A hearty stew made with fermented soybean paste (doenjang), tofu, vegetables, and often meat or seafood. Has a deep, earthy flavor.",
        "ingredients": ["doenjang", "tofu", "zucchini", "potatoes", "mushrooms", "onions", "chili peppers"],
        "cooking_method": "Vegetables and protein are simmered in a broth flavored with doenjang.",
        "cultural_note": "A fundamental dish in Korean cuisine, representing traditional fermented flavors."
    },
    "Doenjang Jjigae": {
        "category": "Stew",
        "description": "A hearty stew made with fermented soybean paste (doenjang), tofu, vegetables, and often meat or seafood.",
        "ingredients": ["doenjang", "tofu", "zucchini", "potatoes", "mushrooms", "onions", "chili peppers"],
        "cooking_method": "Vegetables and protein are simmered in a broth flavored with doenjang.",
        "cultural_note": "A fundamental dish in Korean cuisine, representing traditional fermented flavors."
    },
    "Soft Tofu Stew": {
        "category": "Stew",
        "description": "A spicy, comforting stew featuring soft, silky tofu in a flavorful broth with vegetables, meat or seafood, and an egg.",
        "ingredients": ["soft tofu", "gochugaru", "seafood or pork", "egg", "vegetables", "garlic"],
        "cooking_method": "Ingredients are simmered together and served bubbling hot in a stone pot.",
        "cultural_note": "Perfect comfort food, often eaten with rice to balance the spiciness."
    },
    "Sundubu Jjigae": {
        "category": "Stew",
        "description": "A spicy, comforting stew featuring soft, silky tofu in a flavorful broth with vegetables, meat or seafood, and an egg.",
        "ingredients": ["soft tofu", "gochugaru", "seafood or pork", "egg", "vegetables", "garlic"],
        "cooking_method": "Ingredients are simmered together and served bubbling hot in a stone pot.",
        "cultural_note": "Perfect comfort food, often eaten with rice to balance the spiciness."
    },
    "Napa Cabbage Kimchi": {
        "category": "Kimchi",
        "description": "The most iconic type of kimchi, made from napa cabbage fermented with a seasoning mix of chili powder, garlic, ginger, and salted seafood.",
        "ingredients": ["napa cabbage", "gochugaru", "garlic", "ginger", "fish sauce", "salted shrimp"],
        "cooking_method": "Cabbage is salted, rinsed, then coated with seasoning paste and fermented.",
        "cultural_note": "A UNESCO cultural heritage item and essential component of Korean meals."
    },
    "Baechu Kimchi": {
        "category": "Kimchi",
        "description": "The most iconic type of kimchi, made from napa cabbage fermented with chili powder, garlic, ginger, and salted seafood.",
        "ingredients": ["napa cabbage", "gochugaru", "garlic", "ginger", "fish sauce", "salted shrimp"],
        "cooking_method": "Cabbage is salted, rinsed, then coated with seasoning paste and fermented.",
        "cultural_note": "A UNESCO cultural heritage item and essential component of Korean meals."
    },
    "Grilled Short Ribs": {
        "category": "Grilled Meat",
        "description": "Beef short ribs marinated in a sweet soy-based sauce and grilled. Known for their tender, flavorful meat.",
        "ingredients": ["beef short ribs", "soy sauce", "sugar", "garlic", "sesame oil", "pear"],
        "cooking_method": "Ribs are marinated and grilled over charcoal or on a grill pan.",
        "cultural_note": "A premium Korean BBQ dish, often served at special occasions."
    },
    "Galbi Gui": {
        "category": "Grilled Meat",
        "description": "Beef short ribs marinated in a sweet soy-based sauce and grilled. Known for their tender, flavorful meat.",
        "ingredients": ["beef short ribs", "soy sauce", "sugar", "garlic", "sesame oil", "pear"],
        "cooking_method": "Ribs are marinated and grilled over charcoal or on a grill pan.",
        "cultural_note": "A premium Korean BBQ dish, often served at special occasions."
    },
    "Glass Noodles with Vegetables": {
        "category": "Side Dish",
        "description": "Stir-fried sweet potato glass noodles with colorful vegetables and meat, seasoned with soy sauce and sesame oil.",
        "ingredients": ["sweet potato noodles", "beef or pork", "vegetables", "soy sauce", "sesame oil", "sugar"],
        "cooking_method": "Noodles and ingredients are stir-fried separately then combined.",
        "cultural_note": "A popular side dish at celebrations and a favorite in Korean lunch boxes."
    },
    "Japchae": {
        "category": "Side Dish",
        "description": "Stir-fried sweet potato glass noodles with colorful vegetables and meat, seasoned with soy sauce and sesame oil.",
        "ingredients": ["sweet potato noodles", "beef or pork", "vegetables", "soy sauce", "sesame oil", "sugar"],
        "cooking_method": "Noodles and ingredients are stir-fried separately then combined.",
        "cultural_note": "A popular side dish at celebrations and a favorite in Korean lunch boxes."
    },
    "Fried Chicken": {
        "category": "Fried Food",
        "description": "Crispy fried chicken with an extra-crunchy coating, often double-fried for maximum crispiness.",
        "ingredients": ["chicken", "flour", "cornstarch", "seasonings"],
        "cooking_method": "Chicken is coated and double-fried to achieve signature crispiness.",
        "cultural_note": "Korean fried chicken is famous worldwide for its incredible crunch. Perfect with beer (chimaek)."
    },
    "Huraideu Chikin": {
        "category": "Fried Food",
        "description": "Crispy fried chicken with an extra-crunchy coating, often double-fried for maximum crispiness.",
        "ingredients": ["chicken", "flour", "cornstarch", "seasonings"],
        "cooking_method": "Chicken is coated and double-fried to achieve signature crispiness.",
        "cultural_note": "Korean fried chicken is famous worldwide for its incredible crunch."
    },
    "Seasoned Fried Chicken": {
        "category": "Fried Food",
        "description": "Crispy fried chicken coated in a sweet, spicy, and sticky sauce made with gochujang and garlic.",
        "ingredients": ["fried chicken", "gochujang", "garlic", "soy sauce", "honey", "sesame seeds"],
        "cooking_method": "Fried chicken is tossed in a flavorful sauce while still hot.",
        "cultural_note": "A popular late-night snack, especially when paired with beer."
    },
    "Yangnyeom Chikin": {
        "category": "Fried Food",
        "description": "Crispy fried chicken coated in a sweet, spicy, and sticky sauce made with gochujang and garlic.",
        "ingredients": ["fried chicken", "gochujang", "garlic", "soy sauce", "honey", "sesame seeds"],
        "cooking_method": "Fried chicken is tossed in a flavorful sauce while still hot.",
        "cultural_note": "A popular late-night snack, especially when paired with beer."
    },
    "Black Bean Noodles": {
        "category": "Noodles",
        "description": "Thick wheat noodles topped with a savory black bean sauce made from chunjang (black bean paste), pork, and vegetables.",
        "ingredients": ["wheat noodles", "chunjang (black bean paste)", "pork", "onions", "zucchini", "potatoes"],
        "cooking_method": "Black bean sauce is stir-fried and served over boiled noodles.",
        "cultural_note": "Originally Chinese-Korean cuisine, now a beloved comfort food often delivered."
    },
    "Jjajangmyeon": {
        "category": "Noodles",
        "description": "Thick wheat noodles topped with a savory black bean sauce made from chunjang, pork, and vegetables.",
        "ingredients": ["wheat noodles", "chunjang (black bean paste)", "pork", "onions", "zucchini", "potatoes"],
        "cooking_method": "Black bean sauce is stir-fried and served over boiled noodles.",
        "cultural_note": "Originally Chinese-Korean cuisine, now a beloved comfort food."
    },
    "Spicy Seafood Noodles": {
        "category": "Noodles",
        "description": "Spicy noodle soup loaded with seafood like shrimp, squid, and mussels in a rich, fiery red broth.",
        "ingredients": ["wheat noodles", "seafood", "gochugaru", "vegetables", "garlic", "chicken broth"],
        "cooking_method": "Seafood and vegetables are cooked in spicy broth, served over noodles.",
        "cultural_note": "Another Chinese-Korean dish, often ordered alongside jjajangmyeon."
    },
    "Jjamppong": {
        "category": "Noodles",
        "description": "Spicy noodle soup loaded with seafood in a rich, fiery red broth.",
        "ingredients": ["wheat noodles", "seafood", "gochugaru", "vegetables", "garlic", "chicken broth"],
        "cooking_method": "Seafood and vegetables are cooked in spicy broth, served over noodles.",
        "cultural_note": "Another Chinese-Korean dish, often ordered alongside jjajangmyeon."
    },
    "Seaweed Soup": {
        "category": "Soup",
        "description": "A light, nutritious soup made with miyeok (seaweed), beef or seafood, and sesame oil. Rich in minerals.",
        "ingredients": ["miyeok (seaweed)", "beef or seafood", "sesame oil", "garlic", "soy sauce"],
        "cooking_method": "Seaweed is sautéed with protein and sesame oil, then simmered in broth.",
        "cultural_note": "Traditionally eaten by new mothers for nutrition and on birthdays."
    },
    "Miyeok Guk": {
        "category": "Soup",
        "description": "A light, nutritious soup made with miyeok (seaweed), beef or seafood, and sesame oil.",
        "ingredients": ["miyeok (seaweed)", "beef or seafood", "sesame oil", "garlic", "soy sauce"],
        "cooking_method": "Seaweed is sautéed with protein and sesame oil, then simmered in broth.",
        "cultural_note": "Traditionally eaten by new mothers and on birthdays."
    },
    "Kimchi Fried Rice": {
        "category": "Rice Dish",
        "description": "Fried rice made with aged kimchi, often topped with a fried egg and served with roasted seaweed.",
        "ingredients": ["rice", "kimchi", "pork or spam", "gochugaru", "sesame oil", "egg"],
        "cooking_method": "Kimchi is stir-fried with rice and meat until slightly crispy.",
        "cultural_note": "A perfect way to use leftover rice and aged kimchi."
    },
    "Kimchi Bokkeumbap": {
        "category": "Rice Dish",
        "description": "Fried rice made with aged kimchi, often topped with a fried egg and served with roasted seaweed.",
        "ingredients": ["rice", "kimchi", "pork or spam", "gochugaru", "sesame oil", "egg"],
        "cooking_method": "Kimchi is stir-fried with rice and meat until slightly crispy.",
        "cultural_note": "A perfect way to use leftover rice and aged kimchi."
    }
}

