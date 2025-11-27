"""
Script to build the Attribute Database for Korean Food Attributes
Supports both manual (default) and automated extraction methods
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.attribute_db import AttributeDatabase, create_default_attribute_db
from extract_attributes import build_attribute_database
import config


def main():
    """Create the attribute database using manual or automated method"""
    parser = argparse.ArgumentParser(description='Build Attribute Database')
    parser.add_argument('--extract', action='store_true',
                        help='Use automated extraction from food descriptions (NLP method)')
    parser.add_argument('--kb-path', type=str, default=config.DB_PATH,
                        help='Path to knowledge base (for extraction method)')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum frequency for attributes (extraction method)')
    parser.add_argument('--output', type=str, default=config.ATTRIBUTE_DB_PATH,
                        help='Output path for attribute database')
    
    args = parser.parse_args()
    
    if args.extract:
        # Automated extraction method (NLP)
        print("Using automated extraction method (NLP/frequency analysis)...")
        build_attribute_database(args.kb_path, args.output, args.min_frequency)
    else:
        # Manual method (default attributes)
        print("=" * 70)
        print("Building Attribute Database (Manual Method)")
        print("=" * 70)
        print("Using predefined default attributes.")
        print("For automated extraction from descriptions, use: --extract")
        print()
        
        # Create attribute database
        db = create_default_attribute_db(args.output)
        
        print(f"\n✓ Attribute database created at: {args.output}")
        print(f"  Total attributes: {len(db.get_all_attributes())}")
        
        # Show some examples
        print("\nSample attributes:")
        sample_attrs = ["Spicy", "Grilled", "Chicken", "Soup", "Crispy"]
        for attr in sample_attrs:
            desc = db.get_attribute(attr)
            if desc:
                print(f"  - {attr}: {desc[:80]}...")
        
        print("\n" + "=" * 70)
        print("Attribute database ready for use!")
        print("=" * 70)


if __name__ == "__main__":
    main()

