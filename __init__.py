"""
ComfyUI Sunra.ai Plugin

Custom nodes for integrating Sunra.ai models with ComfyUI, including
FLUX.1 Kontext models and Seedance for advanced AI generation.
"""

from .sunra_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optional: Add web directory for client-side extensions
WEB_DIRECTORY = "./js"

# Export required mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'] 