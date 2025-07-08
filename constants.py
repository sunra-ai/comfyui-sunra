"""
Constants and type mappings for Sunra ComfyUI nodes.

This module consolidates all constants used across the codebase.
"""

# API Configuration
DEFAULT_API_URL = "https://api.sunra.ai/v1"
API_TIMEOUT = 300  # 5 minutes for long operations
USER_AGENT = "comfyui-sunra/1.0.0"

# Type mappings from schema to ComfyUI
SCHEMA_TO_COMFY_TYPE_MAP = {
    "string": "STRING",
    "integer": "INT",
    "float": "FLOAT",
    "boolean": "BOOLEAN",
    "image": "IMAGE",
    "mask": "MASK",
    "audio": "AUDIO",
    "video": "STRING",  # Video URLs
}

# Type mappings from OpenAPI to schema format
OPENAPI_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "float",
    "boolean": "boolean",
}

# Media parameter patterns for type detection
IMAGE_PARAMS = [
    "image",
    "input_image",
    "reference_image",
    "mask",
    "style_image",
    "content_image",
]
VIDEO_PARAMS = ["video", "input_video", "reference_video"]
AUDIO_PARAMS = ["audio", "input_audio", "voice", "music"]

# Multiline text field patterns
MULTILINE_PATTERNS = ["prompt", "description", "text", "caption", "instructions"]

# Default values by type
TYPE_DEFAULTS = {
    "string": "",
    "integer": 0,
    "float": 0.0,
    "boolean": False,
    "image": None,
    "mask": None,
    "audio": None,
    "video": None,
}

# ComfyUI specific constants
FORCE_RERUN_WIDGET = (
    "BOOLEAN",
    {"default": False, "label_off": "No", "label_on": "Yes"},
)

# Image processing
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "webp", "gif"]
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
