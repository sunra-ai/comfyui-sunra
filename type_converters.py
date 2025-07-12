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

# Try to import torchaudio for audio processing
try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

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


# Cache for processed audio data, keyed by URL
_audio_cache: Dict[str, Dict[str, Any]] = {}


def _process_and_cache_audio(audio_url: str) -> Dict[str, Any]:
    """
    Helper to download, process, and cache audio data.
    Avoids redundant downloads and processing for the same URL.
    """
    if audio_url in _audio_cache:
        return _audio_cache[audio_url]
    
    from pathlib import Path
    from datetime import datetime
    import folder_paths
    
    # Create output directory following ComfyUI conventions
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(folder_paths.get_output_directory()) / "SunraAudio" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL or generate one
    filename = audio_url.split("/")[-1].split("?")[0]
    if not filename.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")):
        filename = f"audio_{hash(audio_url)}.mp3"
    
    output_path = output_dir / filename
    
    # Download audio
    try:
        response = requests.get(audio_url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download audio from {audio_url}: {str(e)}")
    
    # Get relative path for ComfyUI
    relative_path = output_path.relative_to(Path(folder_paths.get_output_directory()))
    
    # Load audio and return as ComfyUI AUDIO dict
    waveform, sample_rate = torch.zeros(1, 1, 44100), 44100
    if TORCHAUDIO_AVAILABLE:
        try:
            loaded_waveform, loaded_sample_rate = torchaudio.load(str(output_path))
            # Add batch dimension if not present
            if len(loaded_waveform.shape) == 2:
                loaded_waveform = loaded_waveform.unsqueeze(0)
            waveform, sample_rate = loaded_waveform, loaded_sample_rate
        except Exception as e:
            print(f"Warning: Failed to load audio with torchaudio: {e}")
    else:
        print("Warning: torchaudio not available. Audio will be returned with empty waveform.")
    
    result = {"waveform": waveform, "sample_rate": sample_rate, "path": str(relative_path)}
    _audio_cache[audio_url] = result
    return result


def process_audio_output(data: Union[str, Dict]) -> Dict[str, Any]:
    """
    Process audio output from Sunra API.
    Downloads audio and returns it as ComfyUI AUDIO type.
    """
    # Extract audio URL from various formats
    audio_url = data
    if isinstance(data, dict):
        audio_url = data.get("url") or data.get("file_url") or data.get("audio") or str(data)
    
    if not isinstance(audio_url, str) or not audio_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid audio URL: {audio_url}")
    
    cached_data = _process_and_cache_audio(audio_url)
    return {"waveform": cached_data["waveform"], "sample_rate": cached_data["sample_rate"]}


def process_audio_path_output(data: Union[str, Dict]) -> str:
    """
    Returns the file path of the processed audio.
    This function is robust and can be called independently.
    """
    # Extract audio URL from various formats
    audio_url = data
    if isinstance(data, dict):
        audio_url = data.get("url") or data.get("file_url") or data.get("audio") or str(data)
    
    if not isinstance(audio_url, str) or not audio_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid audio URL: {audio_url}")
    
    cached_data = _process_and_cache_audio(audio_url)
    return cached_data["path"]


def process_3d_output(data: Union[str, Dict]) -> str:
    """
    Process 3D model output from Sunra API.
    Downloads 3D model files and returns path following ComfyUI conventions.
    """
    import zipfile
    import tarfile
    from pathlib import Path
    from datetime import datetime
    import folder_paths

    print(f"Processing 3D model output: {data}")
    
    # Extract model URL from various formats
    model_url = None
    if isinstance(data, str):
        model_url = data
    elif isinstance(data, dict):
        # First check for direct URL fields
        model_url = data.get("url") or data.get("file_url")
        
        # If not found, check for nested SunraFile structures
        if not model_url:
            # Check for model field (Hunyuan3D v2 Turbo and v2.1)
            if "model" in data and isinstance(data["model"], dict):
                model_url = data["model"].get("url")
            # Also check model_archive for v2.1
            elif "model_archive" in data and isinstance(data["model_archive"], dict):
                model_url = data["model_archive"].get("url")
    
    if not isinstance(model_url, str) or not model_url.startswith(("http://", "https://")):
        raise ValueError(f"Unable to get model file path. Could not find a valid model URL in the response: {data}")
    
    # Create output directory following ComfyUI conventions
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(folder_paths.get_output_directory()) / "Sunra3D" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = model_url.split("/")[-1].split("?")[0]
    
    # Check if it's an archive or direct model file
    is_archive = filename.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    
    if is_archive:
        # Download archive
        archive_path = output_dir / filename
        try:
            response = requests.get(model_url, timeout=120, stream=True)
            response.raise_for_status()
            
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download 3D model archive from {model_url}: {str(e)}")
        
        # Extract archive
        extract_dir = output_dir / f"{filename}_extracted"
        extract_dir.mkdir(exist_ok=True)
        
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filename.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        # Find 3D model files in extracted directory
        model_extensions = [".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl"]
        model_files = []
        for ext in model_extensions:
            model_files.extend(extract_dir.rglob(f"*{ext}"))
        
        if model_files:
            # Return the first found model file
            # Return path relative to ComfyUI output folder  
            relative_path = model_files[0].relative_to(Path(folder_paths.get_output_directory()))
            return str(relative_path)
        else:
            # No model file found in archive
            raise RuntimeError(f"No 3D model files found in archive from {model_url}")
    else:
        # Direct model file download
        if not filename.endswith((".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl")):
            filename = f"model_{hash(model_url)}.glb"
        
        output_path = output_dir / filename
        
        try:
            response = requests.get(model_url, timeout=120, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download 3D model from {model_url}: {str(e)}")
        
        # Return path relative to ComfyUI output folder
        relative_path = output_path.relative_to(Path(folder_paths.get_output_directory()))
        return str(relative_path)


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
