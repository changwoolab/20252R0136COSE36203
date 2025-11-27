# CLIP-Based Korean Food Classification Pipeline

## Overview

The CLIP-based pipeline is the core component of the Korean Food Explanation System. It leverages OpenAI's CLIP (Contrastive Language-Image Pre-training) model to classify Korean food images using a vision-language approach. The pipeline supports both standard classification (150 food categories from the training dataset) and zero-shot classification (10+ unseen food categories), making it highly flexible and extensible.

## Architecture

The CLIP-based pipeline consists of four main stages:

```
Input Image → CLIP Classifier → Knowledge Retrieval → Text Generation → Explanation
```

### 1. CLIP Classifier (`src/classifier.py`)

The `KoreanFoodClassifier` class encapsulates the CLIP model and provides:

- **Standard Classification**: Classify images against a fixed set of 150 Korean food categories (from knowledge base)
- **Zero-Shot Classification**: Classify images against 160 classes (10 zero-shot candidates + 150 KB classes)
- **Attribute Prediction**: Predict food attributes (e.g., "Spicy", "Grilled", "Chicken") from images
- **Fine-Tuning Support**: Load and use fine-tuned CLIP models
- **WiSE-FT Ensembling**: Weight-space ensembling with zero-shot model for improved performance

### 2. Knowledge Retrieval (`src/pipeline.py`)

The `KoreanFoodPipeline` orchestrates the complete pipeline:

- **Exact Match Retrieval**: Fast O(1) lookup in knowledge base for in-distribution foods
- **Zero-Shot Handling**: For foods not in KB, retrieves context from similar KB foods
- **Attribute-Aware Retrieval**: Use CLIP to match images to foods based on visual attributes
- **Attribute Database Integration**: Predict attributes and retrieve descriptions for unknown foods

### 3. Text Generation (`src/text_generator.py`)

Generates natural language explanations:

- **Template-Based**: Fast, deterministic explanations (SimpleFoodExplainer)
- **LLM-Based**: Natural language using TinyLlama or similar models (FoodExplainer)
- **Zero-Shot Generation**: Special handling for zero-shot predictions with KB context

### 4. Knowledge Base (`src/knowledge_base.py`)

Contains information about 150 Korean food categories:
- English and Korean names
- Descriptions and categories
- Ingredients and cooking methods
- Cultural notes

## Data Structure

### Training Data (`dataset/`)
- 150 food categories with multiple images per category
- Used for fine-tuning the CLIP model
- All categories have entries in the knowledge base

### Zero-Shot Test Data (`zeroshot_dataset/`)
- 10 food categories NOT in the training data or knowledge base
- Used to evaluate zero-shot generalization capabilities
- Categories: Budae Jjigae, Beef Tartare Bibimbap, Haemul Pajeon, Sujebi, cheonggukjang, Nakji Bokkeum, Agujjim, Hotteok, Patbingsu, Tangsuyuk

## Key Features

### 1. Multiple Prompt Templates with Ensembling

Instead of using a single prompt template, the classifier uses **6 different prompt templates** and ensembles them for better accuracy:

```python
prompt_templates = [
    "a photo of {food}",
    "a picture of {food}",
    "an image of {food}",
    "{food}",
    "Korean food {food}",
    "a dish of {food}"
]
```

**How it works:**
1. Compute text embeddings for each prompt template
2. Average the embeddings across all templates
3. Re-normalize the averaged features
4. Use the ensembled features for classification

This approach improves robustness by capturing different linguistic variations of how food names might be described.

### 2. Temperature Scaling

The classifier applies **temperature scaling** to similarity scores before softmax:

```python
temperature = 0.1  # Lower temperature = sharper distribution
scaled_similarity = similarity / temperature
probs = torch.nn.functional.softmax(scaled_similarity, dim=0)
```

**Why temperature scaling?**
- CLIP outputs cosine similarities in the range [-1, 1]
- Without scaling, softmax probabilities can be too uniform
- Lower temperature (0.1) creates sharper probability distributions
- Makes the classifier more confident in its predictions

### 3. Zero-Shot Classification with Combined Classes

The pipeline supports **true zero-shot classification** - classifying images against both known (KB) and unknown (candidate) food names:

```python
# Zero-shot mode combines:
# - 10 zero-shot candidate foods (from zero_shot_candidate_foods.txt)
# - 150 KB food classes (from knowledge base)
# = 160 total classes for classification

predictions = classifier.classify_image_zero_shot(
    image_path="food.jpg",
    candidate_foods=combined_foods,  # 160 classes
    top_k=5
)
```

**How it works:**
1. Load 10 candidate foods from `zero_shot_candidate_foods.txt`
2. Combine with 150 KB classes (removing duplicates)
3. Compute text embeddings on-the-fly for all 160 classes
4. Compute cosine similarity between image and text embeddings
5. Apply temperature scaling and softmax
6. Return top-k predictions

**Benefits:**
- Tests both in-distribution accuracy (150 KB classes) and zero-shot generalization (10 new classes)
- No need to precompute features for zero-shot candidates
- Can classify against foods not in the original training set

### 4. Zero-Shot Prediction Handling

When a zero-shot food is predicted (not in KB), the pipeline:

**Step 1: Detect Zero-Shot Prediction**
```python
is_zero_shot_prediction = (top_food_name in candidate_foods) and (top_food_name not in kb_food_names)
```

**Step 2: Retrieve Similar KB Foods for Context**
```python
# Look at other predictions to find similar foods in KB
for pred_name, pred_conf in predictions[1:]:
    if pred_name in kb_food_names:
        similar_info = knowledge_base.get_food_info(pred_name)
        similar_kb_foods.append({
            'name': pred_name,
            'confidence': pred_conf,
            'info': similar_info
        })
```

**Step 3: Infer Category and Ingredients**
- Category is inferred from similar KB foods (e.g., if similar foods are "Stews", infer "Stew")
- Ingredients are collected from similar dishes

**Step 4: Generate Explanation with Context**
```python
# LLM or template generates explanation using:
# - Predicted food name (e.g., "Budae Jjigae")
# - Similar KB foods (e.g., "Kimchi Stew", "Soybean Paste Stew")
# - Inferred category (e.g., "Stew")
# - Similar ingredients (e.g., "tofu, pork belly, doenjang")
```

### 5. Attribute-Aware Retrieval

When the predicted food is not found in the knowledge base (for non-zero-shot foods), the pipeline uses **attribute-aware retrieval**:

**Step 1: Predict Attributes**
```python
predicted_attributes = classifier.predict_attributes(
    image_path,
    attribute_list=["Spicy", "Grilled", "Chicken", "Soup", ...],
    top_k=5
)
```

**Step 2: Attribute Database Lookup**
- Retrieve attribute descriptions from the attribute database
- Create a synthetic food_info dict with predicted attributes

**Step 3: Attribute-Aware Matching**
```python
# Weighted combination of label similarity and attribute similarity
combined_score = (1.0 - attribute_weight) * label_score + attribute_weight * attr_score
```

### 6. Fine-Tuning Support

The pipeline supports loading fine-tuned CLIP models:

```python
classifier = KoreanFoodClassifier(
    model_name="openai/clip-vit-base-patch32",  # Base model
    model_path="./models/clip_improved"  # Fine-tuned model path
)
```

**Fine-tuned models include:**
- Model weights
- Processor/tokenizer
- Class mappings (`class_mappings.json`)

### 7. WiSE-FT (Weight-Space Ensembling)

The classifier supports **WiSE-FT** (Weight-Space Ensembling for Fine-Tuning) to prevent catastrophic forgetting:

```python
classifier.ensemble_with_zeroshot(
    zero_shot_model_name="openai/clip-vit-base-patch32",
    alpha=0.7  # 70% fine-tuned, 30% zero-shot
)
```

**Formula:**
```
θ_final = α · θ_fine-tuned + (1 - α) · θ_zero-shot
```

**Benefits:**
- Prevents overfitting to training data
- Maintains zero-shot capabilities
- Improves generalization to unseen foods

## Pipeline Flow

### Standard Classification Flow (No `--zero-shot` flag)

```
1. Load Image
   ↓
2. Extract Image Features (CLIP Vision Encoder)
   ↓
3. Compute Similarity with Precomputed Text Features (150 KB classes)
   ↓
4. Apply Temperature Scaling + Softmax
   ↓
5. Get Top-K Predictions
   ↓
6. Lookup in Knowledge Base (Exact Match)
   ↓
7. If not found → Attribute-Aware Retrieval
   ↓
8. Generate Explanation (LLM/Template)
```

### Zero-Shot Classification Flow (`--zero-shot` flag)

```
1. Load Image
   ↓
2. Load Candidate Foods (10) + KB Classes (150) = 160 Classes
   ↓
3. Extract Image Features (CLIP Vision Encoder)
   ↓
4. Compute Text Features On-the-Fly for All 160 Classes
   ↓
5. Compute Similarity
   ↓
6. Apply Temperature Scaling + Softmax
   ↓
7. Get Top-K Predictions
   ↓
8. Check if Top Prediction is Zero-Shot (not in KB)
   ↓
   ├── If Zero-Shot:
   │   ↓
   │   9a. Find Similar KB Foods from Other Predictions
   │   ↓
   │   10a. Infer Category/Ingredients from Similar Foods
   │   ↓
   │   11a. Generate Explanation with KB Context
   │
   └── If In-Distribution (in KB):
       ↓
       9b. Exact Match Lookup in Knowledge Base
       ↓
       10b. Generate Standard Explanation
```

## Usage Examples

### Standard Classification

```bash
python demo.py --mode single --image "food.jpg"
```

```python
from src.pipeline import create_pipeline

pipeline = create_pipeline(
    knowledge_base_path="food_knowledge_base.json",
    classifier_type="clip"
)

result = pipeline.analyze_food_image("food.jpg", top_k=5)
print(result['identified_food'])
print(result['explanation'])
```

### Zero-Shot Classification

```bash
python demo.py --mode single --image "food.jpg" --zero-shot
```

```python
# Load candidate foods
with open('zero_shot_candidate_foods.txt', 'r') as f:
    candidate_foods = [line.strip() for line in f if line.strip()]

result = pipeline.analyze_food_image(
    "food.jpg",
    candidate_foods=candidate_foods,
    top_k=5
)

# Result for zero-shot food includes similar KB context
print(result['identified_food'])  # e.g., "Budae Jjigae"
print(result['category'])  # Inferred from similar foods: "Stew"
print(result['explanation'])  # Generated with KB context
```

### Evaluating Zero-Shot Performance

```bash
# Evaluate pretrained CLIP model
python evaluate_zeroshot.py

# Evaluate fine-tuned model
python evaluate_zeroshot.py --model-path ./models/clip_improved
```

### Using LLM for Generation

```bash
python demo.py --mode single --image "food.jpg" --zero-shot --use-llm
```

## Configuration Files

### `zero_shot_candidate_foods.txt`
List of 10 food names for zero-shot evaluation:
```
Beef Tartare Bibimbap
Budae Jjigae
Haemul Pajeon
Sujebi
cheonggukjang
Nakji Bokkeum
Agujjim
Hotteok
Patbingsu
Tangsuyuk
```

### `food_knowledge_base.json`
Knowledge base with 150 food entries containing:
- `english_name`, `korean_name`
- `description`, `category`
- `ingredients`, `cooking_method`
- `cultural_note`

### `attribute_db.json`
Attribute database for unknown food handling:
- Flavor attributes (Spicy, Sweet, Savory, etc.)
- Cooking method attributes (Grilled, Fried, Steamed, etc.)
- Ingredient attributes (Beef, Pork, Seafood, etc.)

## Technical Details

### Text Feature Computation

**Standard Classification:**
- Text features are precomputed and cached when `set_food_classes()` is called
- Uses 6 prompt templates with ensembling
- Features are normalized (L2 norm = 1)

**Zero-Shot Classification:**
- Text features are computed on-the-fly for all 160 classes
- Same prompt template ensembling approach
- No caching (computed fresh each time)

### Image Feature Extraction

```python
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

### Similarity Computation

```python
similarity = (image_features @ text_features.T).squeeze(0)
scaled_similarity = similarity / temperature  # temperature = 0.1
probs = torch.nn.functional.softmax(scaled_similarity, dim=0)
```

### Confidence Thresholds

- **CLIP**: Default threshold = 0.001 (very low probabilities with 160 classes)
- **CNN/ViT**: Default threshold = 0.01 (higher confidence)

## Performance Considerations

### Memory Usage

- **Standard Classification**: Precomputed text features (150 × 512 = ~300KB)
- **Zero-Shot Classification**: On-the-fly computation for 160 classes (~320KB)

### Speed

- **Standard Classification**: Fast (precomputed features)
- **Zero-Shot Classification**: Slightly slower (on-the-fly computation)
- **LLM Generation**: Slower but more natural explanations

### Expected Accuracy

| Mode | In-Distribution (150 classes) | Zero-Shot (10 classes) |
|------|-------------------------------|------------------------|
| Pretrained CLIP | ~36% Top-1 | ~36% Top-1 |
| Fine-tuned CLIP | Higher | May vary |
| WiSE-FT | Balanced | Preserved |

## Limitations

1. **Zero-Shot Performance**: Some foods may be confused with visually similar KB foods
2. **Knowledge Base Coverage**: Only 150 foods have detailed information
3. **Attribute Prediction**: Secondary CLIP may not always predict correct attributes
4. **Context Inference**: Zero-shot explanations rely on similar KB foods which may not be accurate

## Future Improvements

1. **Expand Knowledge Base**: Add more food categories
2. **Better Zero-Shot Context**: Use semantic similarity for KB matching
3. **Multi-Modal Retrieval**: Combine image and text for better matching
4. **Active Learning**: Improve fine-tuning with user feedback
5. **Few-Shot Learning**: Support few-shot classification for new foods

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020): Learning Transferable Visual Models From Natural Language Supervision
- [WiSE-FT Paper](https://arxiv.org/abs/2109.01903): Robust Fine-tuning of Zero-shot Models
- [HuggingFace CLIP](https://huggingface.co/docs/transformers/model_doc/clip): CLIP Model Documentation
