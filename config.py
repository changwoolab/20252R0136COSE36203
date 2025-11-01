"""
Configuration file for Korean Food Explanation System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "kfood_dataset")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "kfood_kor_eng_match.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "food_knowledge_base.json")

# Model configurations
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Tiny LLM for generation

# Training configurations
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
IMAGE_SIZE = 224
NUM_WORKERS = 4

# Classification threshold (CLIP outputs low confidence scores even for correct predictions)
CONFIDENCE_THRESHOLD = 0.001

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

