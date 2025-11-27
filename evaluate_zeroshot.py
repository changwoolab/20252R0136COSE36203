"""
Evaluate zero-shot performance of CLIP-based classifier on zeroshot_dataset
Uses pretrained CLIP model (not fine-tuned) to evaluate zero-shot capabilities
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import KoreanFoodClassifier
from src.knowledge_base import FoodKnowledgeBase
import config


def load_zeroshot_candidate_foods(candidate_file_path):
    """
    Load zero-shot candidate food names from text file
    
    Args:
        candidate_file_path: Path to zero_shot_candidate_foods.txt file
        
    Returns:
        List of food names (strings)
    """
    candidate_foods = []
    
    if not os.path.exists(candidate_file_path):
        print(f"Warning: Candidate food file not found at {candidate_file_path}")
        return candidate_foods
    
    with open(candidate_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                candidate_foods.append(line)
    
    return candidate_foods


def collect_zeroshot_samples(zeroshot_dataset_dir):
    """Collect all test samples from zeroshot_dataset"""
    dataset_path = Path(zeroshot_dataset_dir)
    test_data = {}
    
    if not dataset_path.exists():
        raise ValueError(f"Zero-shot dataset directory not found: {zeroshot_dataset_dir}")
    
    # Collect all images from each food class directory
    for food_dir in sorted(dataset_path.iterdir()):
        if food_dir.is_dir():
            food_name = food_dir.name
            # Support both .jpg and .png images
            images = list(food_dir.glob("*.jpg")) + list(food_dir.glob("*.png"))
            
            if len(images) > 0:
                test_data[food_name] = [str(img) for img in sorted(images)]
                print(f"  Found {len(images)} images for {food_name}")
    
    return test_data


def evaluate_zeroshot_classifier(classifier, test_data, is_finetuned=False):
    """Evaluate classifier on test data"""
    print("\n" + "=" * 70)
    if is_finetuned:
        print("Evaluating Fine-Tuned CLIP Classifier")
    else:
        print("Evaluating Zero-Shot CLIP Classifier")
    print("=" * 70)
    
    results = {
        'correct': 0,
        'total': 0,
        'top5_correct': 0,
        'per_class': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    print(f"\nEvaluating on {len(test_data)} classes...")
    print("-" * 70)
    
    # Determine if we should use zero-shot classification (like demo.py)
    # When candidate foods are used, we should use classify_image_zero_shot to match demo.py behavior
    use_zero_shot_method = False
    candidate_foods_list = None
    
    # Check if classifier has food_classes set (from set_food_classes or loaded from model)
    if hasattr(classifier, 'food_classes') and classifier.food_classes:
        # Use zero-shot method to match demo.py behavior exactly
        use_zero_shot_method = True
        candidate_foods_list = classifier.food_classes
        print(f"Using zero-shot classification method with {len(candidate_foods_list)} candidate foods")
        print("(This matches demo.py behavior when candidate_foods are provided)")
    
    for true_label, image_paths in sorted(test_data.items()):
        print(f"Testing {true_label}... ({len(image_paths)} images)")
        
        for image_path in image_paths:
            try:
                # Use classify_image_zero_shot to match demo.py behavior exactly
                # This ensures the same computation path as when candidate_foods are passed to pipeline
                if use_zero_shot_method and candidate_foods_list:
                    predictions = classifier.classify_image_zero_shot(
                        image_path,
                        candidate_foods=candidate_foods_list,
                        top_k=5
                    )
                else:
                    # Fallback to standard classification
                    predictions = classifier.classify_image(image_path, top_k=5)
                
                # Check top-1
                pred_label = predictions[0][0]
                pred_confidence = predictions[0][1]
                
                if pred_label == true_label:
                    results['correct'] += 1
                    results['per_class'][true_label]['correct'] += 1
                    status = "✓"
                else:
                    status = "✗"
                
                # Check top-5
                top5_labels = [pred[0] for pred in predictions]
                if true_label in top5_labels:
                    results['top5_correct'] += 1
                
                results['total'] += 1
                results['per_class'][true_label]['total'] += 1
                
                # Print prediction for each image (optional, can be verbose)
                # print(f"    {status} Predicted: {pred_label} (conf: {pred_confidence:.4f})")
            
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Calculate metrics
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    top5_accuracy = results['top5_correct'] / results['total'] if results['total'] > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    if is_finetuned:
        print("FINE-TUNED MODEL EVALUATION RESULTS")
    else:
        print("ZERO-SHOT EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {results['total']}")
    print(f"Number of classes: {len(test_data)}")
    print(f"\nTop-1 Accuracy: {accuracy:.2%} ({results['correct']}/{results['total']})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2%} ({results['top5_correct']}/{results['total']})")
    
    # Per-class accuracy
    print("\n" + "-" * 70)
    print("Per-Class Accuracy:")
    print("-" * 70)
    class_accs = []
    for class_name, stats in sorted(results['per_class'].items()):
        if stats['total'] > 0:
            class_acc = stats['correct'] / stats['total']
            class_accs.append((class_name, class_acc, stats['correct'], stats['total']))
            print(f"  {class_name:<30} {class_acc:>6.2%} ({stats['correct']}/{stats['total']})")
    
    # Summary statistics
    if class_accs:
        avg_class_acc = sum(acc for _, acc, _, _ in class_accs) / len(class_accs)
        print("\n" + "-" * 70)
        print(f"Average per-class accuracy: {avg_class_acc:.2%}")
        print(f"Best class: {max(class_accs, key=lambda x: x[1])[0]} ({max(class_accs, key=lambda x: x[1])[1]:.2%})")
        print(f"Worst class: {min(class_accs, key=lambda x: x[1])[0]} ({min(class_accs, key=lambda x: x[1])[1]:.2%})")
    
    print("=" * 70)
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': results['total'],
        'num_classes': len(test_data),
        'per_class': dict(results['per_class']),
        'average_per_class_accuracy': avg_class_acc if class_accs else 0.0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Zero-Shot CLIP Classifier')
    parser.add_argument(
        '--zeroshot-dataset',
        type=str,
        default=None,
        help='Path to zeroshot_dataset directory (default: ./zeroshot_dataset)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to fine-tuned CLIP model directory (e.g., ./models/clip_improved). If provided, this overrides --model-name'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='Pretrained CLIP model name (default: openai/clip-vit-base-patch32). Used only if --model-path is not provided'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON)'
    )
    parser.add_argument(
        '--use-all-classes',
        action='store_true',
        help='Use all classes from knowledge base instead of zero-shot candidate foods'
    )
    parser.add_argument(
        '--use-dataset-classes',
        action='store_true',
        help='Use only classes present in zeroshot_dataset (default: use zero-shot candidate foods)'
    )
    parser.add_argument(
        '--candidate-foods-file',
        type=str,
        default=None,
        help='Path to zero_shot_candidate_foods.txt file (default: ./zero_shot_candidate_foods.txt)'
    )
    parser.add_argument(
        '--use-model-classes',
        action='store_true',
        help='If fine-tuned model has saved food classes, use them instead of evaluation configuration'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Zero-Shot CLIP Classifier Evaluation")
    print("=" * 70)
    
    # Determine zeroshot_dataset path
    if args.zeroshot_dataset:
        zeroshot_dataset_dir = args.zeroshot_dataset
    else:
        # Default to ./zeroshot_dataset
        zeroshot_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zeroshot_dataset')
    
    # Check zeroshot_dataset
    if not os.path.exists(zeroshot_dataset_dir):
        print(f"Error: Zero-shot dataset not found at {zeroshot_dataset_dir}")
        sys.exit(1)
    
    # Load knowledge base
    print("\nLoading knowledge base...")
    kb = FoodKnowledgeBase(config.DB_PATH)
    all_food_names = kb.get_food_names()
    print(f"Knowledge base contains {len(all_food_names)} food categories")
    
    # Collect test samples from zeroshot_dataset
    print(f"\nCollecting test samples from: {zeroshot_dataset_dir}")
    test_data = collect_zeroshot_samples(zeroshot_dataset_dir)
    total_samples = sum(len(imgs) for imgs in test_data.values())
    print(f"Collected {total_samples} test samples from {len(test_data)} classes")
    
    # Determine which food classes to use
    if args.use_all_classes:
        # Use all classes from knowledge base
        food_classes = all_food_names
        print(f"\nUsing all {len(food_classes)} classes from knowledge base")
    elif args.use_dataset_classes:
        # Use only classes present in zeroshot_dataset
        food_classes = list(test_data.keys())
        print(f"\nUsing {len(food_classes)} classes from zeroshot_dataset")
    else:
        # Default: Combine zero-shot candidate foods with all knowledge base classes
        # This matches demo.py behavior - uses both candidate foods and KB classes
        if args.candidate_foods_file:
            candidate_file_path = args.candidate_foods_file
        else:
            # Default to ./zero_shot_candidate_foods.txt
            candidate_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zero_shot_candidate_foods.txt')
        
        print(f"\nLoading zero-shot candidate foods from: {candidate_file_path}")
        candidate_foods = load_zeroshot_candidate_foods(candidate_file_path)
        
        if len(candidate_foods) == 0:
            print("Warning: No candidate foods loaded. Falling back to zeroshot_dataset classes.")
            food_classes = list(test_data.keys())
        else:
            # Combine candidate foods with all knowledge base classes
            # Remove duplicates while preserving order (candidate foods first, then KB classes)
            combined_foods = []
            seen = set()
            
            # Add candidate foods first
            for food in candidate_foods:
                if food not in seen:
                    combined_foods.append(food)
                    seen.add(food)
            
            # Add knowledge base classes that aren't already in candidate foods
            kb_added = 0
            for food in all_food_names:
                if food not in seen:
                    combined_foods.append(food)
                    seen.add(food)
                    kb_added += 1
            
            food_classes = combined_foods
            print(f"Loaded {len(candidate_foods)} zero-shot candidate foods")
            print(f"Combined with {kb_added} additional classes from knowledge base")
            print(f"Total: {len(food_classes)} food classes for classification")
    
    # Initialize CLIP classifier
    if args.model_path:
        print(f"\nInitializing fine-tuned CLIP model from: {args.model_path}")
        classifier = KoreanFoodClassifier(model_name=args.model_name, model_path=args.model_path)
        
        # If model already has food classes loaded, optionally use them
        if classifier.food_classes and args.use_model_classes:
            print(f"Fine-tuned model has {len(classifier.food_classes)} food classes loaded")
            print(f"Using {len(classifier.food_classes)} food classes from fine-tuned model")
            food_classes = classifier.food_classes
        elif classifier.food_classes:
            print(f"Fine-tuned model has {len(classifier.food_classes)} food classes loaded")
            print(f"Using {len(food_classes)} food classes from evaluation configuration (use --use-model-classes to use model's classes)")
            classifier.set_food_classes(food_classes)
        else:
            classifier.set_food_classes(food_classes)
    else:
        print(f"\nInitializing pretrained CLIP model: {args.model_name}")
        print("(This is a zero-shot evaluation - no fine-tuning)")
        classifier = KoreanFoodClassifier(model_name=args.model_name)
        classifier.set_food_classes(food_classes)
    
    # Evaluate
    is_finetuned = args.model_path is not None
    results = evaluate_zeroshot_classifier(classifier, test_data, is_finetuned=is_finetuned)
    
    # Save results
    if args.output:
        results['model_name'] = args.model_name
        results['model_path'] = args.model_path if args.model_path else None
        results['is_finetuned'] = args.model_path is not None
        results['zeroshot_dataset_dir'] = zeroshot_dataset_dir
        results['num_food_classes'] = len(food_classes)
        results['food_class_source'] = 'all_kb' if args.use_all_classes else ('dataset' if args.use_dataset_classes else 'candidate_foods')
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

