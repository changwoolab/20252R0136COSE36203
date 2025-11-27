"""
Demo script to showcase the Korean Food Explanation System
"""
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import create_pipeline
import config


def get_random_food_images(num_samples=5):
    """Get random food images from the dataset"""
    dataset_dir = Path(config.DATASET_DIR)
    
    # Get all food categories
    food_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    # Sample random categories
    sampled_dirs = random.sample(food_dirs, min(num_samples, len(food_dirs)))
    
    images = []
    for food_dir in sampled_dirs:
        # Get images from this food category
        image_files = list(food_dir.glob("*.jpg")) + list(food_dir.glob("*.png"))
        if image_files:
            # Pick a random image
            images.append(str(random.choice(image_files)))
    
    return images


def load_candidate_foods(file_path: str) -> list:
    """Load candidate foods from text file"""
    foods = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                foods.append(line)
    return foods


def demo_single_image(pipeline, image_path, candidate_foods=None):
    """Demo with a single image"""
    print("\n" + "=" * 70)
    print(f"Analyzing: {image_path}")
    if candidate_foods:
        print(f"Zero-Shot Mode: Classifying against {len(candidate_foods)} candidate foods")
    print("=" * 70)
    
    result = pipeline.analyze_food_image(
        image_path, 
        top_k=3, 
        confidence_threshold=None,  # Auto-detect
        candidate_foods=candidate_foods
    )
    
    print(pipeline.format_result_text(result))
    
    return result


def demo_batch(pipeline, image_paths, candidate_foods=None):
    """Demo with multiple images"""
    print("\n" + "=" * 70)
    print(f"Batch Analysis: {len(image_paths)} images")
    if candidate_foods:
        print(f"Zero-Shot Mode: Classifying against {len(candidate_foods)} candidate foods")
    print("=" * 70)
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {Path(image_path).name}")
        result = pipeline.analyze_food_image(
            image_path, 
            top_k=1, 
            confidence_threshold=None,  # Auto-detect
            candidate_foods=candidate_foods
        )
        
        if result['success']:
            print(f"  ✓ Identified: {result['identified_food']} ({result.get('korean_name', 'N/A')})")
            print(f"  Confidence: {result['confidence']:.2%}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")


def interactive_demo(pipeline):
    """Interactive demo"""
    print("\n" + "=" * 70)
    print("Interactive Korean Food Explorer")
    print("=" * 70)
    print("\nCommands:")
    print("  1. Enter image path to analyze")
    print("  2. Type 'random' to analyze a random image from dataset")
    print("  3. Type 'list' to see all available foods")
    print("  4. Type 'info <food_name>' to get info about a specific food")
    print("  5. Type 'zero-shot' to use zero-shot mode with candidate foods")
    print("  6. Type 'quit' to exit")
    print()
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif cmd.lower() == 'random':
                images = get_random_food_images(1)
                if images:
                    demo_single_image(pipeline, images[0])
                else:
                    print("No images found in dataset")
            
            elif cmd.lower() == 'list':
                foods = pipeline.list_available_foods()
                print(f"\nAvailable foods ({len(foods)}):")
                for i, food in enumerate(sorted(foods), 1):
                    print(f"  {i}. {food}")
            
            elif cmd.lower().startswith('info '):
                food_name = cmd[5:].strip()
                info = pipeline.get_food_info(food_name)
                if info:
                    print(f"\n{info['english_name']} ({info['korean_name']})")
                    print(f"Category: {info['category']}")
                    print(f"Description: {info['description']}")
                else:
                    print(f"Food '{food_name}' not found")
            
            elif cmd.lower() == 'zero-shot':
                # Zero-shot mode with candidate foods
                candidate_file = "zero_shot_candidate_foods.txt"
                if os.path.exists(candidate_file):
                    candidate_foods = load_candidate_foods(candidate_file)
                    print(f"\n✓ Loaded {len(candidate_foods)} candidate foods from {candidate_file}")
                    print("Enter image path to analyze with zero-shot classification:")
                    img_path = input("  Image path: ").strip()
                    if os.path.exists(img_path):
                        demo_single_image(pipeline, img_path, candidate_foods=candidate_foods)
                    else:
                        print(f"Error: Image not found: {img_path}")
                else:
                    print(f"Error: {candidate_file} not found!")
                    print("Please ensure the file exists in the current directory.")
            
            elif os.path.exists(cmd):
                # Check if candidate foods are available for zero-shot
                candidate_foods = getattr(pipeline, '_candidate_foods', None)
                demo_single_image(pipeline, cmd, candidate_foods=candidate_foods)
            
            else:
                print("Unknown command or file not found")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Korean Food Explanation System Demo')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'batch', 'interactive'],
        default='interactive',
        help='Demo mode'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Image path for single mode'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples for batch mode'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['clip', 'cnn', 'vit'],
        default='clip',
        help='Type of classifier to use'
    )
    parser.add_argument(
        '--cnn-model-path',
        type=str,
        help='Path to trained CNN model'
    )
    parser.add_argument(
        '--vit-model-path',
        type=str,
        help='Path to trained ViT model'
    )
    parser.add_argument(
        '--clip-model-path',
        type=str,
        help='Path to fine-tuned CLIP model'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM for text generation'
    )
    parser.add_argument(
        '--zero-shot',
        action='store_true',
        help='Use zero-shot mode with candidate foods from zero_shot_candidate_foods.txt'
    )
    parser.add_argument(
        '--candidate-foods',
        type=str,
        default='zero_shot_candidate_foods.txt',
        help='Path to candidate foods file for zero-shot mode'
    )
    
    args = parser.parse_args()
    
    # Load candidate foods if zero-shot mode is enabled
    candidate_foods = None
    if args.zero_shot:
        if args.classifier != 'clip':
            print("Error: Zero-shot mode only works with CLIP classifier")
            sys.exit(1)
        
        if os.path.exists(args.candidate_foods):
            candidate_foods = load_candidate_foods(args.candidate_foods)
            print(f"✓ Loaded {len(candidate_foods)} candidate foods for zero-shot classification")
        else:
            print(f"Warning: Candidate foods file not found: {args.candidate_foods}")
            print("Zero-shot mode will be disabled.")
            args.zero_shot = False
    
    # Check if knowledge base exists (optional for zero-shot mode)
    kb_path = config.DB_PATH if os.path.exists(config.DB_PATH) else None
    if not args.zero_shot and not kb_path:
        print(f"Error: Knowledge base not found at {config.DB_PATH}")
        print("Please run 'python build_database.py' first")
        print("Or use --zero-shot mode with --candidate-foods")
        sys.exit(1)
    
    print("=" * 70)
    print("Korean Food Explanation System - Demo")
    if args.zero_shot:
        print("  (Zero-Shot Mode - Using Candidate Foods)")
    print("=" * 70)
    
    # Create pipeline
    import config as cfg
    pipeline = create_pipeline(
        knowledge_base_path=kb_path,
        classifier_type=args.classifier,
        cnn_model_path=args.cnn_model_path,
        vit_model_path=args.vit_model_path,
        clip_model_path=args.clip_model_path,
        use_llm=args.use_llm,
        attribute_db_path=cfg.ATTRIBUTE_DB_PATH if hasattr(cfg, 'ATTRIBUTE_DB_PATH') else None
    )
    
    # Store candidate foods in pipeline for interactive mode
    if candidate_foods:
        pipeline._candidate_foods = candidate_foods
    
    # Run demo based on mode
    if args.mode == 'single':
        if not args.image:
            # Get random image
            images = get_random_food_images(1)
            if images:
                args.image = images[0]
            else:
                print("No images found and no image specified")
                sys.exit(1)
        
        demo_single_image(pipeline, args.image, candidate_foods=candidate_foods)
    
    elif args.mode == 'batch':
        images = get_random_food_images(args.num_samples)
        demo_batch(pipeline, images, candidate_foods=candidate_foods)
    
    elif args.mode == 'interactive':
        interactive_demo(pipeline)


if __name__ == "__main__":
    main()

