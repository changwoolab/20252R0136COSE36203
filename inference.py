"""
Inference Script for Korean Food Explanation System
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import create_pipeline
import config


def main():
    parser = argparse.ArgumentParser(
        description='Korean Food Explanation System - Identify and explain Korean food from images'
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to the food image'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Path to save the result (optional)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (text or json)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.001,
        help='Minimum confidence threshold'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['clip', 'cnn', 'vit'],
        default='clip',
        help='Type of classifier to use (clip, cnn, or vit)'
    )
    parser.add_argument(
        '--cnn-model-type',
        type=str,
        default='resnet50',
        help='CNN model type (resnet50, resnet101, efficientnet_b0-b7, mobilenet_v2)'
    )
    parser.add_argument(
        '--cnn-model-path',
        type=str,
        help='Path to trained CNN model (for CNN classifier)'
    )
    parser.add_argument(
        '--vit-model-type',
        type=str,
        default='vit_base_patch16_224',
        help='ViT model type (vit_tiny/small/base/large_patch16_224)'
    )
    parser.add_argument(
        '--vit-model-path',
        type=str,
        help='Path to trained ViT model (for ViT classifier)'
    )
    parser.add_argument(
        '--clip-model-path',
        type=str,
        help='Path to fine-tuned CLIP model (for CLIP classifier)'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM for text generation (slower but more natural)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='LLM model to use for generation'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Check if knowledge base exists
    if not os.path.exists(config.DB_PATH):
        print(f"Error: Knowledge base not found at {config.DB_PATH}")
        print("Please run 'python build_database.py' first to create the knowledge base")
        sys.exit(1)
    
    print("=" * 70)
    print("Korean Food Explanation System")
    print("=" * 70)
    print(f"\n📸 Analyzing image: {args.image}\n")
    
    # Create pipeline
    try:
        pipeline = create_pipeline(
            knowledge_base_path=config.DB_PATH,
            classifier_type=args.classifier,
            cnn_model_type=args.cnn_model_type,
            cnn_model_path=args.cnn_model_path,
            vit_model_type=args.vit_model_type,
            vit_model_path=args.vit_model_path,
            clip_model_path=args.clip_model_path,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            attribute_db_path=config.ATTRIBUTE_DB_PATH if hasattr(config, 'ATTRIBUTE_DB_PATH') else None
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Analyze image
    try:
        result = pipeline.analyze_food_image(
            args.image,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display result
    if args.format == 'json':
        output = pipeline.format_result_json(result)
    else:
        output = pipeline.format_result_text(result)
    
    print(output)
    
    # Save result if requested
    if args.output:
        pipeline.save_result(result, args.output, format=args.format)
    
    return result


if __name__ == "__main__":
    main()

