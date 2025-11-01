# Korean Food Explanation System - Project Summary

## 🎯 Project Goal

Build a tiny LLM-based pipeline that can:
1. Classify Korean food from images
2. Extract food descriptions from a knowledge base
3. Generate explanations about Korean food in English

## ✅ What Was Built

### Core Components

1. **Knowledge Base System** (`src/knowledge_base.py`)
   - Manages 150 Korean food descriptions
   - Stores: English/Korean names, categories, ingredients, cooking methods, cultural notes
   - JSON-based storage for easy extension

2. **CLIP-based Classifier** (`src/classifier.py`)
   - Uses OpenAI's CLIP for zero-shot food classification
   - Supports 150 Korean food categories
   - Achieves reasonable accuracy without fine-tuning
   - Can be fine-tuned for better performance

3. **Text Generator** (`src/text_generator.py`)
   - Two modes: Template-based (fast) and LLM-based (natural)
   - LLM mode uses TinyLLaMA (1.1B parameters)
   - Generates informative explanations with cultural context

4. **Integrated Pipeline** (`src/pipeline.py`)
   - Combines all components into a unified system
   - Simple API for image analysis
   - Formats results as text or JSON

### Tools and Scripts

1. **Database Builder** (`build_database.py`)
   - Generates descriptions for all 150 Korean foods
   - Creates `food_knowledge_base.json`
   - Includes detailed information for popular dishes

2. **Inference Script** (`inference.py`)
   - CLI tool for analyzing individual images
   - Supports various output formats
   - Configurable parameters (top-k, confidence, LLM usage)

3. **Interactive Demo** (`demo.py`)
   - Three modes: single, batch, interactive
   - Browse Korean foods
   - Test with random images
   - User-friendly interface

4. **Test Suite** (`test_pipeline.py`)
   - Validates all components
   - Tests with real images
   - Ensures system is working correctly

5. **Evaluation Script** (`evaluate.py`)
   - Measures classifier performance
   - Per-class accuracy metrics
   - Configurable test samples

6. **Training Script** (`train_classifier.py`)
   - Optional fine-tuning of CLIP
   - Custom dataset loader
   - Training/validation split

### Documentation

1. **README.md**: Comprehensive project documentation
2. **USAGE.md**: Detailed usage guide with examples
3. **PROJECT_SUMMARY.md**: This file

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Food Image                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              CLIP Classifier (Vision Model)                  │
│  • Encodes image to embedding                                │
│  • Compares with 150 food text embeddings                    │
│  • Outputs ranked predictions with confidence                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Knowledge Base (JSON Database)                    │
│  • Retrieves food information                                │
│  • English/Korean names                                      │
│  • Ingredients, cooking method, cultural notes               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Generator (Template or LLM)                │
│  • Formats information into explanation                      │
│  • Template: Fast, structured                                │
│  • LLM: Natural language, slower                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Output: Detailed Explanation in English            │
│  • Food name (English & Korean)                              │
│  • Description and category                                  │
│  • Ingredients and cooking method                            │
│  • Cultural significance                                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
hansik_clip/
├── src/
│   ├── __init__.py               # Package initialization
│   ├── classifier.py             # CLIP-based food classifier
│   ├── knowledge_base.py         # Knowledge base manager + descriptions
│   ├── text_generator.py         # Text generation (template/LLM)
│   └── pipeline.py               # Main pipeline integration
│
├── dataset/
│   ├── kfood_dataset/            # 150 Korean food categories (images)
│   └── kfood_kor_eng_match.csv   # Korean-English name mappings
│
├── models/                        # Saved models (optional)
│
├── build_database.py             # Generate knowledge base
├── inference.py                  # CLI inference tool
├── demo.py                       # Interactive demo
├── test_pipeline.py              # Test suite
├── evaluate.py                   # Evaluation script
├── train_classifier.py           # Training script
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
│
├── food_knowledge_base.json      # Generated knowledge base (150 entries)
│
├── README.md                     # Main documentation
├── USAGE.md                      # Usage guide
└── PROJECT_SUMMARY.md            # This file
```

## 🚀 Quick Start

```bash
# 1. Build knowledge base
python build_database.py

# 2. Test system
python test_pipeline.py

# 3. Try it out
python demo.py --mode interactive

# 4. Analyze an image
python inference.py --image path/to/food.jpg
```

## 🔑 Key Features

### 1. Zero-Shot Classification
- Works without training on Korean food
- Uses CLIP's pre-trained vision-language capabilities
- Reasonable accuracy out-of-the-box

### 2. Rich Knowledge Base
- 150 Korean foods with detailed information
- Includes 30+ curated descriptions for popular dishes
- Automatic generation for remaining dishes
- Easy to extend with more information

### 3. Flexible Text Generation
- **Template mode** (default): Fast, consistent, structured
- **LLM mode** (optional): Natural language, conversational
- Choice depends on speed vs. quality preference

### 4. Easy to Use
- Simple CLI tools
- Python API for integration
- Interactive demo for exploration
- Comprehensive documentation

### 5. Modular Design
- Each component can be used independently
- Easy to replace or upgrade components
- Well-documented code

## 📈 Performance Characteristics

### Speed
- **Template mode**: ~0.5-1s per image (CPU)
- **LLM mode**: ~2-3s per image (CPU)
- **GPU**: 2-5x faster

### Accuracy
- Good for visually distinct dishes (bibimbap, gimbap, fried chicken)
- Challenging for similar-looking foods (various soups/stews)
- Top-5 accuracy is significantly better than top-1
- Can be improved with fine-tuning

### Coverage
- 150 Korean food categories
- Mix of traditional and modern dishes
- Includes: rice dishes, soups, stews, grilled meats, side dishes, etc.

## 🎓 Technical Highlights

### 1. CLIP for Zero-Shot Learning
- Leverages vision-language pre-training
- No need for Korean food-specific training initially
- Text prompts: "a photo of {food_name}"

### 2. Efficient Knowledge Retrieval
- JSON-based database (fast, simple)
- Could be upgraded to vector DB for semantic search
- Pre-computed embeddings for fast inference

### 3. Lightweight LLM
- TinyLLaMA: Only 1.1B parameters
- Runs on CPU (though slower)
- Good balance of size and quality

### 4. Extensible Design
- Easy to add new foods
- Can swap models (CLIP variants, different LLMs)
- Modular architecture

## 🔬 Potential Improvements

### Short-term
1. Add more curated descriptions for all 150 foods
2. Fine-tune CLIP on Korean food dataset
3. Optimize inference speed
4. Add multi-language support (Korean output)

### Medium-term
1. Web interface (Gradio/Streamlit)
2. Mobile app integration
3. Recipe generation
4. Nutritional information

### Long-term
1. Real-time video food recognition
2. Multi-food detection in single image
3. Restaurant recommendation system
4. Dietary preference filtering (vegetarian, halal, etc.)

## 🎯 Use Cases

1. **Tourism**: Help tourists identify Korean food
2. **Education**: Learn about Korean cuisine
3. **Restaurants**: Menu assistance system
4. **Food Delivery**: Automated food identification
5. **Cultural Exchange**: Introduce Korean food internationally

## 📊 Dataset Information

- **Source**: K-Food dataset
- **Categories**: 150 Korean foods (+ 1 "Pizza")
- **Images per category**: Varies (dozens to hundreds)
- **Format**: JPG images organized by folder
- **Labels**: English names (folders) + Korean names (CSV)

## 💡 Lessons Learned

1. **CLIP is powerful**: Zero-shot classification works surprisingly well
2. **Confidence calibration**: CLIP outputs low confidence scores even when correct
3. **Template vs LLM**: Templates are often sufficient for structured information
4. **Knowledge base quality**: Good descriptions make a big difference
5. **User experience**: Interactive demo helps users understand capabilities

## 🙏 Acknowledgments

- **K-Food Dataset**: For providing comprehensive Korean food images
- **OpenAI CLIP**: For enabling zero-shot vision-language understanding
- **TinyLLaMA**: For lightweight but capable language generation
- **Hugging Face**: For easy model access and usage

## 📝 Notes

- System works on both CPU and GPU
- Designed for ease of use and extensibility
- Balances accuracy, speed, and simplicity
- Can be deployed as-is or improved further

## ✨ Conclusion

This project successfully delivers a complete pipeline for Korean food identification and explanation. It combines modern computer vision (CLIP), knowledge management, and natural language generation into an easy-to-use system. The modular design allows for future enhancements while the current implementation is already functional and useful.

**Status**: ✅ Complete and working
**Tested**: ✅ All components verified
**Documented**: ✅ Comprehensive guides provided
**Ready to use**: ✅ Yes!

---

*Built with ❤️ for Korean food lovers and learners*

