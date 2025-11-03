# Korean Food Explanation System (Hansik CLIP)

A comprehensive AI pipeline for identifying Korean food from images and generating detailed explanations in English.

## 🌟 Features

- **151 Korean Food Categories**: Comprehensive coverage of traditional and popular Korean dishes
- **Multi-modal Classification**: Uses OpenAI's CLIP for zero-shot food recognition
- **Rich Knowledge Base**: Detailed descriptions including ingredients, cooking methods, and cultural context
- **Flexible Text Generation**: Template-based (fast) or LLM-based (natural) explanations
- **Easy to Use**: Simple CLI tools and interactive demo

## 🏗️ Pipeline Architecture

The system consists of three main components:

1. **Food Classification** (CLIP): Identifies Korean food from images using vision-language models
2. **Knowledge Retrieval**: Extracts detailed information from a curated database
3. **Text Generation**: Creates natural language explanations using templates or TinyLLaMA

```
Input Image → CLIP Classifier → Knowledge Base → Text Generator → Explanation
```

## 📁 Project Structure

```
hansik_clip/
├── dataset/                    # Korean food image dataset
│   ├── kfood_dataset/         # 151 food categories with images
│   └── kfood_kor_eng_match.csv # Korean-English name mappings
├── models/                     # Saved model checkpoints (optional)
├── src/                        # Source code
│   ├── classifier.py          # CLIP-based food classifier
│   ├── knowledge_base.py      # Food description database
│   ├── text_generator.py     # Text generation (template/LLM)
│   └── pipeline.py            # Main pipeline integration
├── build_database.py          # Create knowledge base
├── inference.py               # Inference CLI tool
├── demo.py                    # Interactive demo
├── test_pipeline.py           # Test suite
├── train_classifier.py        # Fine-tune CLIP (optional)
├── evaluate.py                # Evaluate classifier
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
└── food_knowledge_base.json   # Generated knowledge base
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Build Knowledge Base

```bash
# Generate descriptions for all 150 Korean foods
python3 build_database.py
```

Output: `food_knowledge_base.json` with 150 food entries

### 3. Test the System

```bash
# Run all tests to verify installation
python3 test_pipeline.py
```

### 4. Try the Demo

```bash
# Interactive demo
python3 demo.py --mode interactive

# Single image demo
python3 demo.py --mode single --image path/to/image.jpg

# Batch demo with 10 random images
python3 demo.py --mode batch --num-samples 10
```

## 💻 Usage Examples

### Basic Inference

```bash
# Analyze a Korean food image
python3 inference.py --image path/to/bibimbap.jpg

# Save result as JSON
python3 inference.py --image path/to/image.jpg --output result.json --format json

# Use LLM for more natural text (slower)
python3 inference.py --image path/to/image.jpg --use-llm

# Show top 5 predictions
python3 inference.py --image path/to/image.jpg --top-k 5
```

### Python API

```python
from src.pipeline import create_pipeline

# Initialize pipeline
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    use_llm=False  # Set to True for LLM-based generation
)

# Analyze an image
result = pipeline.analyze_food_image('path/to/image.jpg', top_k=3)

print(f"Identified: {result['identified_food']}")
print(f"Korean name: {result['korean_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nExplanation:\n{result['explanation']}")

# Get information about a specific food
info = pipeline.get_food_info('Bibimbap')
explanation = pipeline.get_food_explanation('Bibimbap')
```

### Evaluation

```bash
# Evaluate classifier on dataset
python3 evaluate.py --samples-per-class 10 --output eval_results.json
```

### Training (Optional)

The system works well with pretrained CLIP in zero-shot mode. However, you can fine-tune it:

```bash
# Fine-tune CLIP on Korean food dataset
python3 train_classifier.py --epochs 10 --batch-size 32 --lr 1e-5
```

## 📊 Example Output

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
seasoned vegetables, beef, a fried egg, and gochujang 
(Korean chili paste). The name literally means 'mixed rice'.

**Key Ingredients:** rice, vegetables (spinach, bean sprouts, 
carrots, mushrooms), beef, egg, gochujang, sesame oil

**Preparation:** Each ingredient is prepared separately and 
arranged over warm rice. Mixed together before eating.

**Cultural Note:** One of the most iconic Korean dishes, 
representing harmony and balance with its colorful ingredients.
============================================================
```

## 🎯 Dataset

The system uses the K-Food dataset with:
- **150 Korean food categories** (151 folders, excluding Pizza)
- **Images organized by food name** (English)
- **Korean-English name mappings** in CSV format

Each food category contains multiple images for training and evaluation.

## 🧠 Models Used

1. **CLIP (Classification)**: `openai/clip-vit-base-patch32`
   - Zero-shot vision-language model
   - Excellent for food recognition without fine-tuning
   
2. **TinyLLaMA (Text Generation)**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (optional)
   - Lightweight language model
   - Generates natural explanations
   - Can be replaced with template-based generation for speed

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Paths
DATASET_DIR = "dataset/kfood_dataset"
DB_PATH = "food_knowledge_base.json"

# Models
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Parameters
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.001
```

## 🔬 Technical Details

### Classification Approach

The system uses CLIP's vision-language capabilities for zero-shot classification:

1. Encode food category names as text: "a photo of {food_name}"
2. Compute text embeddings for all 150 categories
3. For input image, compute image embedding
4. Calculate cosine similarity between image and all text embeddings
5. Apply softmax to get probability distribution

### Knowledge Base Structure

Each food entry contains:
- English and Korean names
- Category (e.g., Rice Dish, Soup, Grilled Meat)
- Description
- Key ingredients
- Cooking method
- Cultural significance

### Text Generation Options

1. **Template-based (Default)**: Fast, consistent formatting
2. **LLM-based (Optional)**: More natural language, slower

## 📈 Performance

On the K-Food dataset (zero-shot CLIP):
- The model achieves reasonable accuracy for visually distinct foods
- Performance varies by food type (soups/stews can be challenging)
- Top-5 accuracy is significantly higher than top-1

## 🤝 Contributing

To add new foods or improve descriptions:
1. Update `src/knowledge_base.py` with new entries
2. Rebuild database: `python3 build_database.py`
3. Test: `python3 test_pipeline.py`

## 📝 Notes

- The system works on CPU but is faster with GPU
- CLIP may output low confidence scores even for correct predictions (this is normal)
- For best results, use clear, well-lit food images
- The knowledge base can be extended with more detailed information

## 🙏 Acknowledgments

- K-Food dataset for Korean food images
- OpenAI CLIP for vision-language models
- TinyLLaMA for lightweight text generation

