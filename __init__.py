"""
ComfyUI Sunra.ai

Custom nodes for integrating Sunra.ai models with ComfyUI.
"""

from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass


# Import the dynamic node system
from .base_node import SunraBaseNode
from .api_client import get_api_client
from .schema_to_node import load_schema_from_file

def load_schemas_and_create_nodes() -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Load all schema files and create corresponding nodes.
    
    Returns:
        Tuple of (node_class_mappings, node_display_name_mappings)
    """
    node_class_mappings = {}
    node_display_name_mappings = {}
    
    # Get the schemas directory
    current_dir = Path(__file__).parent
    schemas_dir = current_dir / "schemas"
    
    if not schemas_dir.exists():
        print(f"Warning: Schemas directory not found at {schemas_dir}")
        return node_class_mappings, node_display_name_mappings
    
    # Get API client
    api_client = get_api_client()
    
    # Load each schema file
    for schema_file in schemas_dir.glob("*.json"):
        try:
            # Load and validate schema
            schema = load_schema_from_file(str(schema_file))
            
            # Create node class
            node_class = SunraBaseNode.create_from_schema(schema, api_client)
            
            # Generate node name (remove hyphens for class name)
            node_name = f"Sunra{schema['model_id'].replace('-', '').title()}"
            
            # Add to mappings
            node_class_mappings[node_name] = node_class
            node_display_name_mappings[node_name] = schema.get(
                "display_name", 
                schema.get("name", node_name)
            )
            
            print(f"Loaded node: {node_name} from {schema_file.name}")
            
        except Exception as e:
            print(f"Error loading schema {schema_file.name}: {str(e)}")
            continue
    
    return node_class_mappings, node_display_name_mappings


# Load dynamic nodes
DYNAMIC_NODE_CLASS_MAPPINGS, DYNAMIC_NODE_DISPLAY_NAME_MAPPINGS = load_schemas_and_create_nodes()

NODE_CLASS_MAPPINGS = { **DYNAMIC_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = { **DYNAMIC_NODE_DISPLAY_NAME_MAPPINGS}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print summary
print(f"\nSunra ComfyUI Nodes loaded:")
print(f"  - Total nodes: {len(NODE_CLASS_MAPPINGS)}") 
