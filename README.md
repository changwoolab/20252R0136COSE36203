# Korean Food Explanation System (Hansik CLIP)

A CLIP-based AI pipeline for identifying Korean food from images and generating natural language explanations.

## Features

- **CLIP-based Classification**: Uses OpenAI's CLIP for vision-language food recognition
- **150 Korean Food Categories**: Knowledge base with detailed descriptions, ingredients, and cultural context
- **Zero-Shot Classification**: Classify images against 160 classes (150 KB + 10 zero-shot candidates)
- **Attribute-Aware Retrieval**: Uses visual attributes to improve matching for unknown foods
- **WiSE-FT Support**: Weight-space ensembling for improved generalization
- **Flexible Text Generation**: Template-based or LLM-based (TinyLlama)

## Pipeline Architecture

```
Input Image → CLIP Classifier → Knowledge Retrieval → Text Generation → Explanation
```

1. **CLIP Classification**: Identifies food from images using vision-language model
2. **Knowledge Retrieval**: Extracts information from curated database (or infers from similar foods)
3. **Text Generation**: Creates natural language explanations

## Project Structure

```
hansik_clip/
├── dataset/                       # Training data
│   └── kfood_dataset/             # 150 food categories with images
├── zeroshot_dataset/              # Zero-shot evaluation data (10 categories)
├── models/                        # Saved model checkpoints
│   ├── clip_improved/             # Fine-tuned CLIP model
│   └── clip_improved_wiseft_*/    # WiSE-FT ensembled models
├── src/
│   ├── classifier.py              # CLIP-based food classifier
│   ├── knowledge_base.py          # Food description database
│   ├── text_generator.py          # Text generation (Template/LLM)
│   └── pipeline.py                # Main pipeline integration
├── demo.py                        # Interactive demo
├── inference.py                   # Inference CLI tool
├── evaluate.py                    # Evaluate classifier on dataset
├── evaluate_zeroshot.py           # Evaluate zero-shot performance
├── train_clip_improved.py         # Fine-tune CLIP with anti-overfitting
├── build_database.py              # Create knowledge base
├── build_attribute_db.py          # Create attribute database
├── food_knowledge_base.json       # Generated knowledge base (150 foods)
├── attribute_database.json        # Attribute database
├── zero_shot_candidate_foods.txt  # 10 zero-shot candidate food names
└── requirements.txt               # Python dependencies
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Build Knowledge Base

```bash
python3 build_database.py
```

### 3. Run Demo

```bash
# Interactive mode
python3 demo.py --mode interactive

# Single image
python3 demo.py --mode single --image path/to/image.jpg

# With zero-shot mode (160 classes)
python3 demo.py --mode single --image path/to/image.jpg --zero-shot

# With LLM for natural explanations
python3 demo.py --mode single --image path/to/image.jpg --zero-shot --use-llm
```

## Usage

### Basic Inference

```bash
# Analyze a food image
python3 inference.py --image path/to/food.jpg

# Save result as JSON
python3 inference.py --image path/to/image.jpg --output result.json --format json

# Use fine-tuned model
python3 inference.py --image path/to/image.jpg --clip-model-path ./models/clip_improved
```

### Zero-Shot Classification

Zero-shot mode classifies against 160 classes (150 KB classes + 10 zero-shot candidates):

```bash
python3 demo.py --mode single --image path/to/image.jpg --zero-shot
```

```python
from src.pipeline import create_pipeline

pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    classifier_type='clip'
)

# Load zero-shot candidate foods
with open('zero_shot_candidate_foods.txt', 'r') as f:
    candidate_foods = [line.strip() for line in f if line.strip()]

result = pipeline.analyze_food_image(
    'path/to/image.jpg',
    candidate_foods=candidate_foods,
    top_k=5
)

print(f"Identified: {result['identified_food']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

### Using Fine-Tuned Models

```python
from src.pipeline import create_pipeline

# Use fine-tuned CLIP model
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    classifier_type='clip',
    clip_model_path='./models/clip_improved'
)

# Use WiSE-FT ensembled model
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    classifier_type='clip',
    clip_model_path='./models/clip_improved_wiseft_alpha0_5'
)
```

## Training

### Fine-tune CLIP

```bash
# Basic fine-tuning with anti-overfitting techniques
python3 train_clip_improved.py \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-5 \
    --weight-decay 0.01 \
    --freeze-epochs 3 \
    --early-stopping 5

# With WiSE-FT (Weight-Space Ensembling)
python3 train_clip_improved.py \
    --epochs 20 \
    --ensemble 0.5
```

**Training Features:**
- Data augmentation (random crops, flips, color jitter)
- Weight decay and gradient clipping
- Learning rate warmup and cosine annealing
- Early stopping
- WiSE-FT for preserving zero-shot capabilities

## Evaluation

```bash
# Evaluate on standard dataset
python3 evaluate.py --samples-per-class 10

# Evaluate zero-shot performance
python3 evaluate_zeroshot.py

# Evaluate fine-tuned model
python3 evaluate_zeroshot.py --model-path ./models/clip_improved
```

## Text Generation Options

### 1. Template-based (Default)
Fast, deterministic formatting using KB information.

### 2. LLM-based (TinyLlama)
Natural language explanations using TinyLlama-1.1B-Chat:
```bash
python3 demo.py --image path/to/image.jpg --use-llm
```

**Prompt Design:**
- **In-distribution foods**: Full KB information + predicted visual attributes
- **Zero-shot foods**: Only food name + similar KB dishes (simpler prompt)

## Example Output

```
============================================================
🍽️  Korean Food Identification Result
============================================================

Identified Food: Bibimbap
Korean Name: 비빔밥
Confidence: 12.34%
Category: Rice Dish

------------------------------------------------------------
📖 Explanation:
------------------------------------------------------------
**Bibimbap** (비빔밥)

**Category:** Rice Dish

**Description:** A vibrant mixed rice dish topped with 
seasoned vegetables, beef, a fried egg, and gochujang.

**Key Ingredients:** rice, vegetables, beef, egg, gochujang

**Preparation:** Each ingredient is prepared separately and 
arranged over warm rice. Mixed together before eating.
============================================================
```

## Technical Details

### CLIP Classification
- **Prompt Ensembling**: Uses 6 prompt templates and averages embeddings
- **Temperature Scaling**: 0.1 for sharper probability distributions
- **Zero-Shot**: Computes text features on-the-fly for candidate foods

### Knowledge Base
- 150 Korean food entries with descriptions, ingredients, and cultural notes
- Attribute strings for attribute-aware retrieval

### Zero-Shot Handling
For foods not in the KB:
1. Find similar KB foods from other predictions
2. Infer category and ingredients from similar foods
3. Generate explanation with KB context

## Configuration

Edit `config.py` to customize paths and model settings:

```python
DATASET_DIR = "dataset/kfood_dataset"
DB_PATH = "food_knowledge_base.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## Notes

- Works on CPU but faster with GPU
- CLIP may output low confidence scores (normal for 160 classes)
- For best results, use clear, well-lit food images

## Acknowledgments

- K-Food dataset for Korean food images
- OpenAI CLIP for vision-language models
- TinyLLaMA for lightweight text generation
