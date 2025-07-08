#!/usr/bin/env python3
"""
Generate schemas for Sunra ComfyUI nodes.

This script provides a command-line interface for generating node schemas
from OpenAPI definitions or manual configurations.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from schema_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_generator import SchemaGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate schemas for Sunra ComfyUI nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all schemas
  python generate_schemas.py
  
  # Generate a specific model schema
  python generate_schemas.py --model flux-kontext-dev
  
  # Dry run to see what would be generated
  python generate_schemas.py --dry-run
  
  # Validate existing schemas
  python generate_schemas.py --validate
  
  # List available models
  python generate_schemas.py --list
  
  # Generate to custom output directory
  python generate_schemas.py --output-dir ../custom_nodes/comfyui-sunra/schemas
"""
    )
    
    parser.add_argument(
        "--output-dir", 
        default="../schemas",
        help="Output directory for schemas (default: ../schemas)"
    )
    
    parser.add_argument(
        "--model",
        help="Generate schema for a specific model"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be generated without saving"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate existing schemas"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached OpenAPI schemas"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = SchemaGenerator(output_dir=args.output_dir)
    
    # Handle different commands
    if args.list:
        print("Available models:")
        for model in generator.list_available_models():
            print(f"  - {model}")
        return
    
    if args.validate:
        stats = generator.validate_schemas()
        # Exit with error code if any schemas are invalid
        sys.exit(0 if stats["invalid"] == 0 else 1)
    
    if args.model:
        # Generate single model
        schema = generator.generate_single_schema(args.model, args.dry_run)
        sys.exit(0 if schema else 1)
    else:
        # Generate all schemas
        stats = generator.generate_all_schemas(args.dry_run)
        sys.exit(0 if stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()