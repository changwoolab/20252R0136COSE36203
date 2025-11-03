# ViT (Vision Transformer) Classifier Integration

## ✅ What Was Added

I've successfully integrated a **ViT-based classifier** as the third classification option. Now you have three choices:

1. **CLIP Classifier**: Zero-shot, vision-language model
2. **CNN Classifier**: ResNet/EfficientNet/MobileNet architectures  
3. **ViT Classifier** (new): Vision Transformer architectures

## 📦 New Files Created

### 1. `src/vit_classifier.py` (350+ lines)
ViT-based food classifier with support for multiple architectures:
- **vit_tiny_patch16_224** (~5.7M params) - Fastest, lightweight
- **vit_small_patch16_224** (~22M params) - Small and efficient
- **vit_base_patch16_224** (~86M params) - Default, good balance
- **vit_large_patch16_224** (~304M params) - Most accurate, slowest

**Key Features**:
- Same interface as CLIP and CNN classifiers
- Uses timm library for pretrained models
- Save/load trained models
- Batch processing support
- Image embeddings extraction

### 2. `train_vit.py` (300+ lines)
Complete training script for ViT classifier:
- Data augmentation optimized for transformers
- Cosine annealing learning rate scheduler
- AdamW optimizer with weight decay
- Checkpoint saving and resume support
- Supports 224x224 and 384x384 image sizes

## 🔧 Modified Files

### 1. `requirements.txt`
- Added `timm>=0.9.0` (PyTorch Image Models library)

### 2. `src/pipeline.py`
- Added ViT classifier support
- Added `vit_model_type` and `vit_model_path` parameters
- Automatic ViT model initialization

### 3. `config.py`
- Added ViT model configurations
- Default ViT model type setting
- ViT model path configuration

### 4. `inference.py`
- Added `--vit-model-type` flag
- Added `--vit-model-path` flag
- Updated `--classifier` choices to include 'vit'

### 5. `demo.py`
- Added `--vit-model-path` flag
- Updated classifier choices to include 'vit'

## 🚀 How to Use

### Option 1: Use CLIP (Default)

```bash
python3 inference.py --image path/to/image.jpg
```

### Option 2: Use CNN

```bash
python3 inference.py --image path/to/image.jpg --classifier cnn
```

### Option 3: Use Untrained ViT

```bash
# Use tiny ViT (fastest)
python3 inference.py --image path/to/image.jpg --classifier vit --vit-model-type vit_tiny_patch16_224

# Use base ViT (default)
python3 inference.py --image path/to/image.jpg --classifier vit

# Use large ViT (most accurate)
python3 inference.py --image path/to/image.jpg --classifier vit --vit-model-type vit_large_patch16_224
```

### Option 4: Train and Use ViT

```bash
# Step 1: Train ViT classifier
python3 train_vit.py --model-type vit_base_patch16_224 --epochs 20 --batch-size 32

# Step 2: Use trained ViT classifier
python3 inference.py --image path/to/image.jpg --classifier vit --vit-model-path models/vit_trained
```

## 🎓 Training the ViT Classifier

### Basic Training

```bash
# Train base ViT (recommended)
python3 train_vit.py --epochs 20 --batch-size 32
```

### Training Different ViT Sizes

```bash
# Tiny ViT (fastest training, good for quick experiments)
python3 train_vit.py --model-type vit_tiny_patch16_224 --epochs 15 --batch-size 64 --lr 0.0001

# Small ViT (good balance)
python3 train_vit.py --model-type vit_small_patch16_224 --epochs 20 --batch-size 48 --lr 0.0001

# Base ViT (default, recommended)
python3 train_vit.py --model-type vit_base_patch16_224 --epochs 25 --batch-size 32 --lr 0.0001

# Large ViT (best accuracy, requires more GPU memory)
python3 train_vit.py --model-type vit_large_patch16_224 --epochs 30 --batch-size 16 --lr 0.00005
```

### Resume Training

```bash
python3 train_vit.py --resume models/vit_trained --epochs 30
```

## 📊 Comparison: CLIP vs CNN vs ViT

### CLIP (Vision-Language Model)
✅ **Pros:**
- Zero-shot learning (works immediately)
- Understands text descriptions
- No training required
- Good generalization

❌ **Cons:**
- Lower accuracy than trained models
- Larger model size
- Slower inference

### CNN (Convolutional Neural Network)
✅ **Pros:**
- Fast inference
- Proven architecture
- Lower memory usage
- Good accuracy when trained

❌ **Cons:**
- Needs training
- Inductive bias (assumes locality)
- Less global context

### ViT (Vision Transformer)
✅ **Pros:**
- State-of-the-art accuracy when trained
- Global attention mechanism
- Better at capturing relationships
- Flexible architecture

❌ **Cons:**
- Requires more training data
- Slower than CNN
- Higher memory usage
- Needs training for best results

## 📈 Model Specifications

| Model | Parameters | Image Size | Speed | Best For |
|-------|-----------|------------|-------|----------|
| ViT-Tiny | 5.7M | 224 | ⚡⚡⚡ | Fast experiments, mobile |
| ViT-Small | 22M | 224 | ⚡⚡ | Good balance |
| ViT-Base | 86M | 224 | ⚡ | Production (default) |
| ViT-Large | 304M | 224 | 🐌 | Maximum accuracy |

## 🎯 When to Use Which

### Use CLIP when:
- You need immediate results without training
- You want to experiment quickly
- You're dealing with new/unseen classes
- You need language understanding

### Use CNN when:
- You need fast inference
- You have limited GPU memory
- You want proven, stable architecture
- Speed is more important than accuracy

### Use ViT when:
- You need maximum accuracy
- You have sufficient training data
- You can afford longer training time
- You have good GPU resources
- You want state-of-the-art performance

## 📝 Python API Examples

### Using ViT with Pipeline

```python
from src.pipeline import create_pipeline

# Use untrained ViT
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    classifier_type='vit',
    vit_model_type='vit_base_patch16_224'
)

# Use trained ViT
pipeline = create_pipeline(
    knowledge_base_path='food_knowledge_base.json',
    classifier_type='vit',
    vit_model_path='models/vit_trained'
)

result = pipeline.analyze_food_image('image.jpg')
```

### Direct ViT Classifier Usage

```python
from src.vit_classifier import ViTFoodClassifier
from src.knowledge_base import FoodKnowledgeBase

# Initialize
kb = FoodKnowledgeBase('food_knowledge_base.json')
classifier = ViTFoodClassifier(
    model_type='vit_base_patch16_224',
    num_classes=150
)
classifier.set_food_classes(kb.get_food_names())

# Classify
predictions = classifier.classify_image('image.jpg', top_k=5)
for food_name, confidence in predictions:
    print(f"{food_name}: {confidence:.2%}")

# Save trained model
classifier.save_model('models/my_vit')

# Load trained model
classifier.load_model('models/my_vit')
```

## 💡 Training Tips

### 1. Learning Rate
- Start with `0.0001` for Base/Small ViT
- Use `0.00005` for Large ViT
- Decrease if loss is unstable

### 2. Batch Size
- ViT-Tiny: 64 (or higher)
- ViT-Small: 48
- ViT-Base: 32
- ViT-Large: 16 (or lower depending on GPU)

### 3. Training Duration
- Minimum 15-20 epochs
- ViT needs more epochs than CNN
- Monitor validation accuracy for convergence

### 4. Data Augmentation
- Training script includes good defaults
- RandomCrop, RandomHorizontalFlip
- ColorJitter for robustness

## 🔬 Technical Details

### ViT Architecture
- Splits image into patches (16x16)
- Treats patches as sequence tokens
- Multi-head self-attention mechanism
- Positional embeddings for spatial info
- Classification token for predictions

### Key Differences from CNN
- No convolutional layers
- Global receptive field from start
- Learns relationships between all patches
- Position-aware through embeddings

### Why ViT for Food Classification?
- Captures global context (whole dish composition)
- Learns spatial relationships (ingredients layout)
- Flexible attention patterns
- State-of-the-art performance on ImageNet

## 📊 Expected Performance (After Training)

### Top-1 Accuracy (estimated):
- **ViT-Tiny**: 75-82%
- **ViT-Small**: 78-85%
- **ViT-Base**: 80-88%
- **ViT-Large**: 82-90%

### Top-5 Accuracy (estimated):
- **ViT-Tiny**: 88-93%
- **ViT-Small**: 90-95%
- **ViT-Base**: 92-96%
- **ViT-Large**: 93-97%

### Inference Speed (CPU):
- **ViT-Tiny**: ~150ms per image
- **ViT-Small**: ~250ms per image
- **ViT-Base**: ~400ms per image
- **ViT-Large**: ~800ms per image

### Inference Speed (GPU):
- **ViT-Tiny**: ~20ms per image
- **ViT-Small**: ~30ms per image
- **ViT-Base**: ~50ms per image
- **ViT-Large**: ~100ms per image

## 🎉 Summary

You now have **three powerful classifiers** to choose from:

1. ✅ **CLIP** - Zero-shot, ready to use
2. ✅ **CNN** - Fast and efficient
3. ✅ **ViT** - State-of-the-art accuracy potential

**Recommendation**:
- **Prototyping**: Start with CLIP
- **Production (speed)**: Train CNN
- **Production (accuracy)**: Train ViT
- **Best overall**: Train ViT-Base or ViT-Small

All three classifiers share the same pipeline interface, making it easy to switch between them and compare performance!

## 🚀 Quick Start Commands

```bash
# Install timm library
pip install timm

# Test with untrained ViT
python3 inference.py --image image.jpg --classifier vit

# Train ViT
python3 train_vit.py --model-type vit_base_patch16_224 --epochs 20

# Use trained ViT
python3 inference.py --image image.jpg --classifier vit --vit-model-path models/vit_trained

# Compare all three classifiers
python3 inference.py --image image.jpg --classifier clip
python3 inference.py --image image.jpg --classifier cnn
python3 inference.py --image image.jpg --classifier vit
```

---

*ViT classifier integration complete! Vision Transformers are ready to use!* 🎊



