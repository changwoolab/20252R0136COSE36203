# Usage Guide - Korean Food Explanation System

## Quick Start

### 1. Setup (One-time)

```bash
# Navigate to project directory
cd /home/aikusrv04/hansik_clip

# Build the knowledge base
python3 build_database.py

# Test the system
python3 test_pipeline.py
```

### 2. Basic Usage

#### Option A: Interactive Demo (Recommended)

```bash
python3 demo.py --mode interactive
```

Commands in interactive mode:
- Type an image path to analyze it
- Type `random` to analyze a random image from dataset
- Type `list` to see all 150 Korean food categories
- Type `info Bibimbap` to get info about a specific food
- Type `quit` to exit

#### Option B: Analyze Single Image

```bash
python3 inference.py --image dataset/kfood_dataset/Bibimbap/Img_072_0001.jpg
```

#### Option C: Batch Analysis

```bash
python3 demo.py --mode batch --num-samples 10
```

## Advanced Usage

### 1. Using with LLM (More Natural Text)

```bash
# This will use TinyLLaMA for text generation (slower, ~2-3 seconds per image)
python3 inference.py --image path/to/image.jpg --use-llm
```

### 2. Save Results

```bash
# Save as text file
python3 inference.py --image path/to/image.jpg --output result.txt --format text

# Save as JSON
python3 inference.py --image path/to/image.jpg --output result.json --format json
```

### 3. Adjust Confidence Threshold

```bash
# Show only high-confidence predictions
python3 inference.py --image path/to/image.jpg --confidence-threshold 0.01

# Show all predictions (very low threshold)
python3 inference.py --image path/to/image.jpg --confidence-threshold 0.0001
```

### 4. Show More Predictions

```bash
# Show top 10 predictions instead of 3
python3 inference.py --image path/to/image.jpg --top-k 10
```

## Python API Usage

### Basic Example

```python
from src.pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    use_llm=False  # Fast template-based generation
)

# Analyze image
result = pipeline.analyze_food_image('path/to/image.jpg')

if result['success']:
    print(f"Food: {result['identified_food']}")
    print(f"Korean: {result['korean_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\n{result['explanation']}")
```

### Advanced Example

```python
from src.pipeline import create_pipeline
from src.classifier import KoreanFoodClassifier
from src.knowledge_base import FoodKnowledgeBase

# 1. Use classifier only
kb = FoodKnowledgeBase('food_knowledge_base.json')
classifier = KoreanFoodClassifier()
classifier.set_food_classes(kb.get_food_names())

predictions = classifier.classify_image('image.jpg', top_k=5)
for food_name, confidence in predictions:
    print(f"{food_name}: {confidence:.2%}")

# 2. Get detailed info about a food
pipeline = create_pipeline('food_knowledge_base.json')
info = pipeline.get_food_info('Bibimbap')
print(f"Category: {info['category']}")
print(f"Description: {info['description']}")
print(f"Ingredients: {', '.join(info['ingredients'])}")

# 3. Generate explanation without image
explanation = pipeline.get_food_explanation('Kimchi Stew')
print(explanation)
```

### Batch Processing

```python
from src.pipeline import create_pipeline
from pathlib import Path

pipeline = create_pipeline('food_knowledge_base.json')

# Process all images in a directory
image_dir = Path('my_food_images')
results = []

for image_path in image_dir.glob('*.jpg'):
    result = pipeline.analyze_food_image(str(image_path))
    results.append({
        'image': image_path.name,
        'food': result['identified_food'],
        'confidence': result['confidence']
    })
    print(f"✓ {image_path.name}: {result['identified_food']}")

# Save summary
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Evaluation and Testing

### Evaluate Classifier Performance

```bash
# Test on 10 images per food category
python3 evaluate.py --samples-per-class 10

# Test on 20 images and save results
python3 evaluate.py --samples-per-class 20 --output evaluation.json
```

### Run Full Test Suite

```bash
python3 test_pipeline.py
```

## Training (Optional)

The system works well with pretrained CLIP in zero-shot mode. Fine-tuning is optional:

```bash
# Fine-tune CLIP on Korean food dataset
python3 train_classifier.py --epochs 10 --batch-size 32 --lr 1e-5 --output-dir models/clip_finetuned

# Use fine-tuned model
python3 inference.py --image path/to/image.jpg --model-path models/clip_finetuned
```

## Troubleshooting

### Issue: "Knowledge base not found"
**Solution**: Run `python3 build_database.py` first

### Issue: "CUDA out of memory"
**Solution**: The system works fine on CPU. If on GPU, reduce batch size in config.py

### Issue: "Low confidence scores"
**Solution**: This is normal with CLIP. Even correct predictions may have ~0.7% confidence. The relative ranking is what matters, not absolute scores.

### Issue: Slow inference
**Solution**: 
- Make sure you're using `use_llm=False` (template-based generation)
- With LLM, expect 2-3 seconds per image on CPU
- Use GPU for faster inference with LLM

## Tips for Best Results

1. **Use clear, well-lit images**: Blurry or dark images reduce accuracy
2. **Center the food**: Make sure the food fills most of the frame
3. **Avoid multiple dishes**: Best results with a single dish
4. **Check alternatives**: Look at top-3 or top-5 predictions, not just top-1
5. **Temperature matters**: Some foods look very similar (e.g., various soups)

## File Locations

- **Knowledge Base**: `food_knowledge_base.json`
- **Dataset**: `dataset/kfood_dataset/`
- **Models** (if trained): `models/`
- **Configuration**: `config.py`

## Common Workflows

### Workflow 1: Identify Unknown Korean Food

1. Take a photo or have an image
2. Run: `python3 inference.py --image photo.jpg`
3. Read the explanation to learn about the dish

### Workflow 2: Browse Korean Foods

1. Run: `python3 demo.py --mode interactive`
2. Type `list` to see all foods
3. Type `info Bibimbap` to learn about specific foods
4. Type `random` to see examples

### Workflow 3: Build a Food Recognition App

```python
from src.pipeline import create_pipeline

# Initialize once (slow)
pipeline = create_pipeline('food_knowledge_base.json')

# Use many times (fast)
def identify_food(image_path):
    result = pipeline.analyze_food_image(image_path)
    return {
        'name': result['identified_food'],
        'korean': result['korean_name'],
        'description': result['detailed_info']['description']
    }
```

## Performance Expectations

- **CPU Inference**: ~0.5-1 second per image (without LLM)
- **CPU with LLM**: ~2-3 seconds per image
- **GPU Inference**: ~0.1-0.2 seconds per image (without LLM)
- **GPU with LLM**: ~0.3-0.5 seconds per image

- **Top-1 Accuracy**: Varies by food type (soups are harder than distinct dishes)
- **Top-5 Accuracy**: Significantly better, usually includes correct answer

## Need Help?

1. Run test suite: `python3 test_pipeline.py`
2. Check this guide
3. Review README.md for technical details

