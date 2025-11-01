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


def demo_single_image(pipeline, image_path):
    """Demo with a single image"""
    print("\n" + "=" * 70)
    print(f"Analyzing: {image_path}")
    print("=" * 70)
    
    result = pipeline.analyze_food_image(image_path, top_k=3, confidence_threshold=0.001)
    
    print(pipeline.format_result_text(result))
    
    return result


def demo_batch(pipeline, image_paths):
    """Demo with multiple images"""
    print("\n" + "=" * 70)
    print(f"Batch Analysis: {len(image_paths)} images")
    print("=" * 70)
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {Path(image_path).name}")
        result = pipeline.analyze_food_image(image_path, top_k=1, confidence_threshold=0.001)
        
        if result['success']:
            print(f"  ✓ Identified: {result['identified_food']} ({result['korean_name']})")
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
    print("  5. Type 'quit' to exit")
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
            
            elif os.path.exists(cmd):
                demo_single_image(pipeline, cmd)
            
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
        '--use-llm',
        action='store_true',
        help='Use LLM for text generation'
    )
    
    args = parser.parse_args()
    
    # Check if knowledge base exists
    if not os.path.exists(config.DB_PATH):
        print(f"Error: Knowledge base not found at {config.DB_PATH}")
        print("Please run 'python build_database.py' first")
        sys.exit(1)
    
    print("=" * 70)
    print("Korean Food Explanation System - Demo")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_pipeline(
        knowledge_base_path=config.DB_PATH,
        use_llm=args.use_llm
    )
    
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
        
        demo_single_image(pipeline, args.image)
    
    elif args.mode == 'batch':
        images = get_random_food_images(args.num_samples)
        demo_batch(pipeline, images)
    
    elif args.mode == 'interactive':
        interactive_demo(pipeline)


if __name__ == "__main__":
    main()

