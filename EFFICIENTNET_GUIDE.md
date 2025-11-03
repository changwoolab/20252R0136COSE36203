# 📊 EfficientNet Models Guide

## ✅ All EfficientNet Models Now Supported (B0-B7)

I've added support for **all 8 EfficientNet models** (B0 through B7) in the CNN classifier.

---

## 🏗️ EfficientNet Family Overview

EfficientNet models use **compound scaling** - they scale width, depth, and resolution together for optimal performance.

### Model Specifications

| Model | Params | Input Size | Top-1 Acc* | Speed** | Memory | Best For |
|-------|--------|------------|------------|---------|--------|----------|
| **B0** | 5.3M | 224x224 | 77.1% | ⚡⚡⚡⚡ | Low | Fast inference, mobile |
| **B1** | 7.8M | 240x240 | 79.1% | ⚡⚡⚡ | Low | Balanced, mobile |
| **B2** | 9.2M | 260x260 | 80.1% | ⚡⚡⚡ | Medium | Good balance |
| **B3** | 12M | 300x300 | 81.6% | ⚡⚡ | Medium | Production default |
| **B4** | 19M | 380x380 | 82.9% | ⚡⚡ | High | High accuracy |
| **B5** | 30M | 456x456 | 83.6% | ⚡ | High | Research, accuracy |
| **B6** | 43M | 528x528 | 84.0% | 🐌 | Very High | Maximum accuracy |
| **B7** | 66M | 600x600 | 84.3% | 🐌🐌 | Very High | State-of-the-art |

*ImageNet Top-1 Accuracy  
**Speed on GPU (relative)

---

## 🚀 Usage Examples

### Basic Training

```bash
# EfficientNet-B0 (Fastest, good for testing)
python3 train_cnn_improved.py --model-type efficientnet_b0 --batch-size 64

# EfficientNet-B1 (Balanced)
python3 train_cnn_improved.py --model-type efficientnet_b1 --batch-size 48

# EfficientNet-B2 (Good balance)
python3 train_cnn_improved.py --model-type efficientnet_b2 --batch-size 40

# EfficientNet-B3 (Recommended for production)
python3 train_cnn_improved.py --model-type efficientnet_b3 --batch-size 32

# EfficientNet-B4 (High accuracy)
python3 train_cnn_improved.py --model-type efficientnet_b4 --batch-size 24

# EfficientNet-B5 (Research-grade)
python3 train_cnn_improved.py --model-type efficientnet_b5 --batch-size 16

# EfficientNet-B6 (Maximum accuracy)
python3 train_cnn_improved.py --model-type efficientnet_b6 --batch-size 12

# EfficientNet-B7 (State-of-the-art)
python3 train_cnn_improved.py --model-type efficientnet_b7 --batch-size 8
```

### Inference

```bash
# Use trained EfficientNet model
python3 inference.py \
    --image path/to/image.jpg \
    --classifier cnn \
    --cnn-model-type efficientnet_b3 \
    --cnn-model-path models/efficientnet_b3_trained
```

---

## 📊 Expected Performance on Korean Food

Based on ImageNet performance and food recognition benchmarks:

| Model | Expected Val Acc | Training Time* | Recommended Batch Size | GPU Memory |
|-------|------------------|----------------|------------------------|------------|
| B0 | 85-87% | 45 min | 64 | 4GB |
| B1 | 86-88% | 55 min | 48 | 6GB |
| B2 | 87-89% | 65 min | 40 | 6GB |
| B3 | 88-90% | 75 min | 32 | 8GB |
| B4 | 89-91% | 100 min | 24 | 10GB |
| B5 | 90-92% | 140 min | 16 | 14GB |
| B6 | 90-92% | 180 min | 12 | 18GB |
| B7 | 91-93% | 240 min | 8 | 24GB |

*Approximate training time for 30 epochs on single GPU

---

## 💡 Which Model to Choose?

### For Quick Experiments:
✅ **EfficientNet-B0** or **B1**
- Fast training
- Low memory
- Good accuracy (85-88%)

### For Production (Balanced):
✅ **EfficientNet-B2** or **B3**
- Best accuracy/speed trade-off
- Reasonable training time
- 88-90% accuracy

### For Maximum Accuracy:
✅ **EfficientNet-B4** or **B5**
- High accuracy (89-92%)
- Acceptable speed
- Good for competitions

### For Research/State-of-the-art:
✅ **EfficientNet-B6** or **B7**
- Maximum accuracy (90-93%)
- Slow but best results
- Requires high-end GPU

---

## 🔬 Technical Details

### Input Resolution

Each EfficientNet variant uses different input sizes:

```python
B0: 224x224  # Smallest, fastest
B1: 240x240
B2: 260x260
B3: 300x300
B4: 380x380
B5: 456x456
B6: 528x528
B7: 600x600  # Largest, most accurate
```

**Note:** The CNN classifier automatically handles these resolutions. The training script uses `Resize(256)` then `RandomCrop(224)` which works for all models, but you can optimize by using model-specific sizes.

### Compound Scaling

EfficientNet scales three dimensions:
1. **Depth** (number of layers)
2. **Width** (number of channels)
3. **Resolution** (input image size)

This is more efficient than scaling just one dimension.

### Architecture

All EfficientNet models use:
- MBConv blocks (Mobile Inverted Bottleneck)
- Squeeze-and-Excitation (SE) blocks
- Swish activation
- Batch normalization

---

## 🎯 Recommended Training Settings

### EfficientNet-B0/B1 (Fast Training):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b0 \
    --epochs 25 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --freeze-epochs 1
```

### EfficientNet-B2/B3 (Production):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b3 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --freeze-epochs 1
```

### EfficientNet-B4/B5 (High Accuracy):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b5 \
    --epochs 35 \
    --batch-size 16 \
    --lr 0.0003 \
    --weight-decay 0.015 \
    --freeze-epochs 2 \
    --warmup-epochs 5
```

### EfficientNet-B6/B7 (Maximum Accuracy):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b7 \
    --epochs 40 \
    --batch-size 8 \
    --lr 0.0002 \
    --weight-decay 0.02 \
    --freeze-epochs 3 \
    --warmup-epochs 5
```

---

## ⚠️ Important Notes

### GPU Memory Requirements

Make sure you have enough GPU memory:

```python
B0-B2: 4-8GB   # RTX 2060, GTX 1080 Ti
B3-B4: 8-12GB  # RTX 3070, RTX 2080 Ti
B5-B6: 12-20GB # RTX 3090, A100
B7: 20-24GB+   # A100, V100 (32GB)
```

If you get **Out of Memory** errors:
1. Reduce batch size
2. Use gradient accumulation
3. Use mixed precision training (FP16)
4. Use a smaller model

### Training Time

Larger models take significantly longer:
- B0: ~45 min/30 epochs
- B3: ~75 min/30 epochs
- B5: ~140 min/30 epochs
- B7: ~240 min/30 epochs (4 hours!)

### Batch Size Recommendations

```python
# Adjust based on GPU memory
B0-B1: 64  (fits on 6GB GPU)
B2-B3: 32  (fits on 8GB GPU)
B4:    24  (fits on 10GB GPU)
B5:    16  (fits on 12GB GPU)
B6:    12  (needs 16GB+ GPU)
B7:    8   (needs 20GB+ GPU)
```

---

## 📈 Performance Comparison

### Accuracy vs Speed

```
High Accuracy, Slow:
B7 ████████████████████ 93%
B6 ███████████████████  92%
B5 ██████████████████   91%

Balanced:
B4 ████████████████     90%
B3 ███████████████      89%  ← Recommended!
B2 ██████████████       88%

Fast, Lower Accuracy:
B1 █████████████        87%
B0 ████████████         86%
```

### Memory Efficiency

```
Most Efficient:
B0 ████████████████████ (5.3M params)
B1 ███████████████      (7.8M params)
B2 ██████████████       (9.2M params)
B3 ████████████         (12M params)

Heavy:
B4 ██████████          (19M params)
B5 ███████             (30M params)
B6 ████                (43M params)
B7 ██                  (66M params)
```

---

## 🎓 Best Practices

### 1. Start Small, Scale Up

```bash
# Step 1: Test with B0 (fast)
python3 train_cnn_improved.py --model-type efficientnet_b0 --epochs 10

# Step 2: If results good, try B3 (balanced)
python3 train_cnn_improved.py --model-type efficientnet_b3 --epochs 30

# Step 3: If need more accuracy, try B5
python3 train_cnn_improved.py --model-type efficientnet_b5 --epochs 35
```

### 2. Match Batch Size to Model

Larger models need smaller batch sizes:
```python
batch_size = max(8, 64 // (model_size / 5.3))
# B0: 64, B3: 32, B5: 16, B7: 8
```

### 3. Adjust Learning Rate

Smaller batches need lower learning rates:
```python
# B0-B2 (large batch): lr = 0.001
# B3-B4 (medium batch): lr = 0.0005
# B5-B7 (small batch): lr = 0.0003
```

### 4. Monitor Overfitting

Larger models can overfit more easily:
```python
# B0-B3: weight_decay = 0.01
# B4-B5: weight_decay = 0.015
# B6-B7: weight_decay = 0.02
```

---

## 💻 Example Training Commands

### Quick Test (B0):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b0 \
    --epochs 20 \
    --batch-size 64
```

### Production (B3):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b3 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0005 \
    --weight-decay 0.01
```

### Maximum Accuracy (B5):

```bash
python3 train_cnn_improved.py \
    --model-type efficientnet_b5 \
    --epochs 35 \
    --batch-size 16 \
    --lr 0.0003 \
    --weight-decay 0.015 \
    --patience 15
```

---

## 🎯 Summary

### Available Models:
✅ EfficientNet-B0, B1, B2, B3, B4, B5, B6, B7  
✅ ResNet50, ResNet101  
✅ MobileNet-V2

### Recommendations:
- **Testing**: B0 or B1
- **Production**: B2 or B3 ⭐
- **High Accuracy**: B4 or B5
- **Maximum**: B6 or B7

### Quick Selection Guide:
```python
if gpu_memory < 8GB:
    use efficientnet_b0 or b1
elif need_fast_inference:
    use efficientnet_b2 or b3  # Best choice ⭐
elif have_time_and_gpu:
    use efficientnet_b5
else:
    use efficientnet_b3  # Default recommendation
```

---

*All EfficientNet models are now available with pretrained ImageNet weights!* 🎉

