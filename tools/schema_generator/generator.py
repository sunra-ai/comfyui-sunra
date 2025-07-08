"""
Schema generator module.

This module provides the main logic for generating schemas from OpenAPI definitions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from .converter import OpenAPIToSchemaConverter
from .models_config import get_all_models, get_model_by_name


class SchemaGenerator:
    """Main schema generator class."""
    
    def __init__(self, output_dir: str = "schemas"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.converter = OpenAPIToSchemaConverter()
    
    def generate_schema(self, model_config: Dict[str, Any], dry_run: bool = False) -> Optional[Dict[str, Any]]:
        """Generate schema for a single model."""
        print(f"Processing {model_config['name']}...")
        
        try:
            # Generate from OpenAPI
            schema = self.converter.generate_schema(
                model_config["openapi_url"],
                model_config["endpoint"],
                model_config.get("overrides")
            )
            
            output_file = self.output_dir / f"{model_config['name']}.json"
            
            if dry_run:
                print(f"  Would save to: {output_file}")
                print(f"  Model ID: {schema['model_id']}")
                print(f"  Inputs: {len(schema['inputs']['required'])} required, "
                      f"{len(schema['inputs']['optional'])} optional")
                print(f"  Outputs: {len(schema['outputs'])}")
            else:
                self.converter.save_schema(schema, str(output_file))
                print(f"  ✓ Saved to: {output_file}")
            
            return schema
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return None
    
    def generate_all_schemas(self, dry_run: bool = False) -> Dict[str, int]:
        """Generate all schemas."""
        sunra_models = get_all_models()
        
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0
        }
        
        print("=== Generating Schemas from OpenAPI ===\n")
        
        # Generate schemas from OpenAPI
        for model_config in sunra_models:
            schema = self.generate_schema(model_config, dry_run)
            stats["total"] += 1
            if schema:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            print()
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def generate_single_schema(self, model_name: str, dry_run: bool = False) -> Optional[Dict[str, Any]]:
        """Generate schema for a single model by name."""
        model_config = get_model_by_name(model_name)
        
        if not model_config:
            print(f"Error: Model '{model_name}' not found")
            return None
        
        return self.generate_schema(model_config, dry_run)
    
    def validate_schemas(self) -> Dict[str, int]:
        """Validate all generated schemas."""
        print("\n=== Validating Schemas ===\n")
        
        stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0
        }
        
        for schema_file in self.output_dir.glob("*.json"):
            print(f"Validating {schema_file.name}...")
            
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                # Check required fields
                required_fields = ["model_id", "name", "api_endpoint", "inputs", "outputs"]
                missing = [field for field in required_fields if field not in schema]
                
                if missing:
                    print(f"  ✗ Missing required fields: {missing}")
                    stats["invalid"] += 1
                else:
                    print(f"  ✓ Valid")
                    stats["valid"] += 1
                
                stats["total"] += 1
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                stats["invalid"] += 1
                stats["total"] += 1
        
        print(f"\nTotal: {stats['total']} schemas")
        print(f"Valid: {stats['valid']}")
        print(f"Invalid: {stats['invalid']}")
        
        return stats
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        sunra_models = get_all_models()
        return sorted([m["name"] for m in sunra_models])
    
    def _print_summary(self, stats: Dict[str, int]):
        """Print generation summary."""
        print("\n=== Summary ===")
        print(f"Total models: {stats['total']}")
        print(f"Successfully generated: {stats['success']}")
        print(f"Failed: {stats['failed']}")