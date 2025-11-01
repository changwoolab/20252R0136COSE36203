"""
Evaluation script for Korean Food Classifier
"""
import os
import sys
from pathlib import Path
import random
from collections import defaultdict
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifier import KoreanFoodClassifier
from src.knowledge_base import FoodKnowledgeBase
import config


def collect_test_samples(dataset_dir, samples_per_class=10):
    """Collect test samples from dataset"""
    dataset_path = Path(dataset_dir)
    test_data = {}
    
    for food_dir in dataset_path.iterdir():
        if food_dir.is_dir():
            food_name = food_dir.name
            images = list(food_dir.glob("*.jpg"))
            
            # Sample images
            if len(images) > samples_per_class:
                sampled = random.sample(images, samples_per_class)
            else:
                sampled = images
            
            test_data[food_name] = [str(img) for img in sampled]
    
    return test_data


def evaluate_classifier(classifier, test_data):
    """Evaluate classifier on test data"""
    print("\nEvaluating classifier...")
    print("=" * 70)
    
    results = {
        'correct': 0,
        'total': 0,
        'top5_correct': 0,
        'per_class': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for true_label, image_paths in test_data.items():
        print(f"Testing {true_label}... ({len(image_paths)} images)")
        
        for image_path in image_paths:
            try:
                predictions = classifier.classify_image(image_path, top_k=5)
                
                # Check top-1
                pred_label = predictions[0][0]
                if pred_label == true_label:
                    results['correct'] += 1
                    results['per_class'][true_label]['correct'] += 1
                
                # Check top-5
                top5_labels = [pred[0] for pred in predictions]
                if true_label in top5_labels:
                    results['top5_correct'] += 1
                
                results['total'] += 1
                results['per_class'][true_label]['total'] += 1
            
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
    
    # Calculate metrics
    accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
    top5_accuracy = results['top5_correct'] / results['total'] if results['total'] > 0 else 0
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total samples: {results['total']}")
    print(f"Top-1 Accuracy: {accuracy:.2%} ({results['correct']}/{results['total']})")
    print(f"Top-5 Accuracy: {top5_accuracy:.2%} ({results['top5_correct']}/{results['total']})")
    
    # Per-class accuracy
    print("\nPer-class accuracy (top 10 worst):")
    class_accs = []
    for class_name, stats in results['per_class'].items():
        if stats['total'] > 0:
            class_acc = stats['correct'] / stats['total']
            class_accs.append((class_name, class_acc, stats['correct'], stats['total']))
    
    # Sort by accuracy
    class_accs.sort(key=lambda x: x[1])
    
    for i, (class_name, acc, correct, total) in enumerate(class_accs[:10], 1):
        print(f"  {i}. {class_name}: {acc:.2%} ({correct}/{total})")
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'total_samples': results['total'],
        'per_class': dict(results['per_class'])
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Korean Food Classifier')
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=10,
        help='Number of samples per class to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save evaluation results (JSON)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Korean Food Classifier Evaluation")
    print("=" * 70)
    
    # Check dataset
    if not os.path.exists(config.DATASET_DIR):
        print(f"Error: Dataset not found at {config.DATASET_DIR}")
        sys.exit(1)
    
    # Load knowledge base
    print("\nLoading knowledge base...")
    kb = FoodKnowledgeBase(config.DB_PATH)
    food_names = kb.get_food_names()
    print(f"Loaded {len(food_names)} food categories")
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = KoreanFoodClassifier()
    classifier.set_food_classes(food_names)
    
    # Collect test samples
    print(f"\nCollecting test samples ({args.samples_per_class} per class)...")
    test_data = collect_test_samples(config.DATASET_DIR, args.samples_per_class)
    total_samples = sum(len(imgs) for imgs in test_data.values())
    print(f"Collected {total_samples} test samples from {len(test_data)} classes")
    
    # Evaluate
    results = evaluate_classifier(classifier, test_data)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()

