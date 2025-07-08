"""
Model configurations for Sunra schema generation.

This module loads model configurations from sunra_models.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_models() -> List[Dict[str, Any]]:
    """Load model configurations from JSON file."""
    models_file = Path(__file__).parent.parent / "sunra_models.json"
    
    if not models_file.exists():
        raise FileNotFoundError(f"Models configuration file not found: {models_file}")
    
    with open(models_file, 'r') as f:
        return json.load(f)


def get_all_models() -> List[Dict[str, Any]]:
    """Get all model configurations."""
    return load_models()


def get_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific model configuration by name."""
    models = load_models()
    for model in models:
        if model["name"] == name:
            return model
    return None