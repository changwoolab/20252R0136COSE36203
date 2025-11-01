"""
Quick test script to verify the pipeline works
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import create_pipeline
import config


def test_knowledge_base():
    """Test knowledge base loading"""
    print("\n[TEST 1] Testing Knowledge Base...")
    try:
        from src.knowledge_base import FoodKnowledgeBase
        kb = FoodKnowledgeBase(config.DB_PATH)
        food_names = kb.get_food_names()
        print(f"✓ Knowledge base loaded: {len(food_names)} foods")
        
        # Test retrieval
        if food_names:
            sample_food = food_names[0]
            info = kb.get_food_info(sample_food)
            print(f"✓ Retrieved info for: {sample_food}")
            print(f"  Korean name: {info['korean_name']}")
            print(f"  Category: {info['category']}")
        
        return True
    except Exception as e:
        print(f"✗ Knowledge base test failed: {e}")
        return False


def test_classifier():
    """Test classifier initialization"""
    print("\n[TEST 2] Testing Classifier...")
    try:
        from src.classifier import KoreanFoodClassifier
        from src.knowledge_base import FoodKnowledgeBase
        
        kb = FoodKnowledgeBase(config.DB_PATH)
        food_names = kb.get_food_names()
        
        classifier = KoreanFoodClassifier()
        classifier.set_food_classes(food_names[:10])  # Test with subset
        print(f"✓ Classifier initialized with {len(food_names[:10])} classes")
        
        return True
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_generator():
    """Test text generator"""
    print("\n[TEST 3] Testing Text Generator...")
    try:
        from src.text_generator import SimpleFoodExplainer
        from src.knowledge_base import FoodKnowledgeBase
        
        kb = FoodKnowledgeBase(config.DB_PATH)
        food_names = kb.get_food_names()
        
        if food_names:
            explainer = SimpleFoodExplainer()
            sample_food = food_names[0]
            info = kb.get_food_info(sample_food)
            
            explanation = explainer.generate_explanation(sample_food, info)
            print(f"✓ Generated explanation for: {sample_food}")
            print(f"  Length: {len(explanation)} characters")
        
        return True
    except Exception as e:
        print(f"✗ Text generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test full pipeline"""
    print("\n[TEST 4] Testing Full Pipeline...")
    try:
        pipeline = create_pipeline(
            knowledge_base_path=config.DB_PATH,
            use_llm=False
        )
        print("✓ Pipeline initialized successfully")
        
        # Test getting food info
        foods = pipeline.list_available_foods()
        print(f"✓ Pipeline can access {len(foods)} foods")
        
        if foods:
            sample_food = foods[0]
            explanation = pipeline.get_food_explanation(sample_food)
            print(f"✓ Generated explanation via pipeline")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image():
    """Test with a real image from dataset"""
    print("\n[TEST 5] Testing with Real Image...")
    try:
        from pathlib import Path
        
        # Find a sample image
        dataset_path = Path(config.DATASET_DIR)
        sample_image = None
        
        for food_dir in dataset_path.iterdir():
            if food_dir.is_dir():
                images = list(food_dir.glob("*.jpg"))
                if images:
                    sample_image = str(images[0])
                    expected_food = food_dir.name
                    break
        
        if not sample_image:
            print("✗ No sample images found in dataset")
            return False
        
        print(f"Using sample image: {sample_image}")
        print(f"Expected food: {expected_food}")
        
        # Create pipeline and analyze
        pipeline = create_pipeline(
            knowledge_base_path=config.DB_PATH,
            use_llm=False
        )
        
        result = pipeline.analyze_food_image(sample_image, top_k=3, confidence_threshold=0.001)
        
        if result['success']:
            print(f"✓ Image analyzed successfully")
            print(f"  Identified: {result['identified_food']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Top 3 predictions:")
            for i, (name, conf) in enumerate(result['predictions'], 1):
                print(f"    {i}. {name} ({conf:.2%})")
            
            return True
        else:
            print(f"✗ Image analysis failed: {result.get('error')}")
            return False
        
    except Exception as e:
        print(f"✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Korean Food Explanation System - Test Suite")
    print("=" * 70)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not os.path.exists(config.DB_PATH):
        print(f"✗ Knowledge base not found at {config.DB_PATH}")
        print("Please run 'python build_database.py' first")
        sys.exit(1)
    
    if not os.path.exists(config.DATASET_DIR):
        print(f"⚠ Dataset not found at {config.DATASET_DIR}")
        print("Some tests may be skipped")
    
    print("✓ Prerequisites check passed")
    
    # Run tests
    results = []
    
    results.append(("Knowledge Base", test_knowledge_base()))
    results.append(("Classifier", test_classifier()))
    results.append(("Text Generator", test_text_generator()))
    results.append(("Full Pipeline", test_pipeline()))
    
    if os.path.exists(config.DATASET_DIR):
        results.append(("Real Image Test", test_with_real_image()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 70)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python demo.py' for an interactive demo")
        print("  2. Run 'python inference.py --image <path>' to analyze an image")
        print("  3. Run 'python evaluate.py' to evaluate classifier performance")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

