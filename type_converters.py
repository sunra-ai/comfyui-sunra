"""
Type conversion utilities for Sunra ComfyUI nodes.

This module handles conversion between Sunra schema types and ComfyUI types,
including data transformation for images, videos, and other media types.
"""

import base64
import io
from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import requests
import torch
from PIL import Image

from .constants import SCHEMA_TO_COMFY_TYPE_MAP

# Try to import sunra_client for image uploads
try:
    import sunra_client

    SUNRA_CLIENT_AVAILABLE = True
except ImportError:
    SUNRA_CLIENT_AVAILABLE = False

# Default configurations for different types
DEFAULT_CONFIGS = {
    "FLOAT": {
        "step": 0.01,
        "round": 0.001,
    },
    "INT": {
        "step": 1,
    },
}


def convert_schema_type_to_comfyui(
    schema_type: str, schema_config: Dict[str, Any]
) -> Tuple[Union[str, List], Dict]:
    """
    Convert a schema type definition to ComfyUI input type.

    Args:
        schema_type: The type from schema (e.g., "string", "integer", "image")
        schema_config: Configuration dictionary from schema

    Returns:
        Tuple of (type, config) for ComfyUI INPUT_TYPES
    """
    # Handle enum types
    if "enum" in schema_config:
        return (
            schema_config["enum"],
            {"default": schema_config.get("default", schema_config["enum"][0])},
        )

    # Get base type
    base_type = SCHEMA_TO_COMFY_TYPE_MAP.get(schema_type, "STRING")

    # Build type configuration
    type_config = {}

    # Add default value if specified
    if "default" in schema_config:
        type_config["default"] = schema_config["default"]

    # Add min/max constraints
    if "min" in schema_config:
        type_config["min"] = schema_config["min"]
    if "max" in schema_config:
        type_config["max"] = schema_config["max"]

    # Add step for numeric types
    if base_type in ["FLOAT", "INT"] and "step" in schema_config:
        type_config["step"] = schema_config["step"]
    elif base_type in DEFAULT_CONFIGS:
        # Use default step if not specified
        type_config.update(DEFAULT_CONFIGS[base_type])

    # Handle multiline strings
    if base_type == "STRING" and schema_config.get("multiline", False):
        type_config["multiline"] = True
        # Add dynamic prompts support for prompt fields
        if "prompt" in schema_config.get("name", "").lower():
            type_config["dynamicPrompts"] = True

    # Handle display names
    if "display_name" in schema_config:
        type_config["display"] = schema_config["display_name"]

    return (base_type, type_config)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to PIL Image (expects ComfyUI BHWC format)."""
    # Handle batch dimension
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch

    # Convert to numpy
    image_np = tensor.cpu().numpy()

    # ComfyUI uses HWC format, no need to transpose

    # Ensure proper value range
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Handle different formats
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 3:  # RGB
            return Image.fromarray(image_np, "RGB")
        elif image_np.shape[2] == 4:  # RGBA
            return Image.fromarray(image_np, "RGBA")
        elif image_np.shape[2] == 1:  # Grayscale
            return Image.fromarray(image_np.squeeze(2), "L")
    elif len(image_np.shape) == 2:  # Grayscale
        return Image.fromarray(image_np, "L")

    raise ValueError(f"Unsupported tensor shape: {image_np.shape}")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor in ComfyUI format."""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array
    image_np = np.array(image).astype(np.float32) / 255.0

    # Add batch dimension (BHWC format for ComfyUI)
    image_np = image_np[None, :, :, :]

    return torch.from_numpy(image_np)


def image_to_base64(
    image: Union[Image.Image, torch.Tensor], format: str = "PNG"
) -> str:
    """Convert image to base64 string."""
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)

    buffered = io.BytesIO()
    image.save(buffered, format=format.upper())
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_base64}"


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))


def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download image from {url}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to process downloaded image: {str(e)}")


def process_image_input(value: Any) -> Optional[str]:
    """
    Process image input for Sunra API.
    Converts various image formats to URL (preferred) or base64 string.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value  # Already a URL or base64 string

    # Convert tensor to PIL Image if needed
    if isinstance(value, torch.Tensor):
        value = tensor_to_pil(value)

    if not isinstance(value, Image.Image):
        raise ValueError(f"Unsupported image input type: {type(value)}")

    # Try to upload image to get URL, fall back to base64
    if SUNRA_CLIENT_AVAILABLE:
        try:
            return sunra_client.upload_image(value, format="png")
        except Exception as e:
            print(f"Warning: Failed to upload image with sunra_client: {e}")

    return image_to_base64(value)


def process_image_output(data: Union[str, Dict, List]) -> torch.Tensor:
    """
    Process image output from Sunra API.
    Converts various formats to tensor.
    """
    if isinstance(data, list):
        # Recursively process each item and concatenate
        tensors = [process_image_output(item) for item in data]
        return torch.cat(tensors, dim=0) if tensors else torch.zeros(1, 3, 512, 512)

    # Extract image data from various formats
    image_data = data
    if isinstance(data, dict):
        image_data = data.get("base64") or data.get("url") or data

    if not isinstance(image_data, str):
        return torch.zeros(1, 3, 512, 512)

    # Convert to PIL Image
    if image_data.startswith("data:image"):
        image = base64_to_image(image_data)
    else:
        image = download_image_from_url(image_data)

    return pil_to_tensor(image)


def process_video_output(data: Union[str, Dict]) -> Any:
    """
    Process video output from Sunra API.
    Downloads video and returns it as ComfyUI VideoFromFile.
    """
    import os
    import tempfile
    from pathlib import Path
    from comfy_api.input_impl.video_types import VideoFromFile
    
    # Extract video URL from various formats
    video_url = data
    if isinstance(data, dict):
        video_url = data.get("url") or data.get("file_url") or str(data)
    
    if not isinstance(video_url, str) or not video_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid video URL: {video_url}")
    
    # Create temp directory for videos if it doesn't exist
    output_dir = Path(tempfile.gettempdir()) / "comfyui_sunra_videos"
    output_dir.mkdir(exist_ok=True)
    
    # Extract filename from URL or generate one
    filename = video_url.split("/")[-1].split("?")[0]
    if not filename.endswith((".mp4", ".webm", ".mov", ".avi")):
        filename = f"video_{hash(video_url)}.mp4"
    
    output_path = output_dir / filename
    
    # Download video if not already cached
    if not output_path.exists():
        try:
            response = requests.get(video_url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download video from {video_url}: {str(e)}")
    
    # Read video file into memory and return as VideoFromFile
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    video_io = io.BytesIO(video_bytes)
    return VideoFromFile(video_io)


def infer_type_from_name(name: str) -> Optional[str]:
    """
    Infer type from parameter name.
    Used as a fallback when type is not explicitly specified.
    """
    name_lower = name.lower()

    # Image-related names
    if any(word in name_lower for word in ["image", "img", "photo", "picture"]):
        return "image"

    # Mask-related names
    if "mask" in name_lower:
        return "mask"

    # Video-related names
    if any(word in name_lower for word in ["video", "movie", "clip"]):
        return "video"

    # Audio-related names
    if any(word in name_lower for word in ["audio", "sound", "music", "voice"]):
        return "audio"

    # Numeric types
    if any(word in name_lower for word in ["count", "number", "steps", "iterations"]):
        return "integer"

    if any(word in name_lower for word in ["scale", "strength", "weight", "ratio"]):
        return "float"

    # Boolean types
    if any(
        word in name_lower
        for word in ["enable", "disable", "use", "is_", "has_", "should_"]
    ):
        return "boolean"

    # Default to string
    return "string"


def validate_value(value: Any, schema_type: str, schema_config: Dict[str, Any]) -> Any:
    """
    Validate and convert a value according to schema definition.
    """
    # Check enum values
    if "enum" in schema_config and value not in schema_config["enum"]:
        raise ValueError(
            f"Value {value} not in allowed values: {schema_config['enum']}"
        )

    # Check min/max constraints
    if "min" in schema_config and value < schema_config["min"]:
        raise ValueError(f"Value {value} is below minimum {schema_config['min']}")

    if "max" in schema_config and value > schema_config["max"]:
        raise ValueError(f"Value {value} is above maximum {schema_config['max']}")

    # Type conversions
    if schema_type == "integer" and isinstance(value, float):
        return int(value)
    elif schema_type == "float" and isinstance(value, int):
        return float(value)
    elif schema_type == "string" and not isinstance(value, str):
        return str(value)
    elif schema_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ["true", "yes", "1", "on"]
        return bool(value)

    return value
