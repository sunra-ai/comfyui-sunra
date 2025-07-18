"""
Type conversion utilities for Sunra ComfyUI nodes.

This module handles conversion between Sunra schema types and ComfyUI types,
including data transformation for images, videos, and other media types.
"""

import base64
import io
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import requests
import torch
from PIL import Image

from .constants import SCHEMA_TO_COMFY_TYPE_MAP

# Try to import sunra_client for file uploads
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

# Try to import folder_paths for ComfyUI paths
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

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

# Cache for processed media data
_media_cache: Dict[str, Dict[str, Any]] = {}


def _upload_to_sunra(data: bytes, content_type: str, filename: str) -> Optional[str]:
    """
    Upload data to Sunra CDN and return URL.
    
    Args:
        data: Binary data to upload
        content_type: MIME type of the data
        filename: Filename for the upload
        
    Returns:
        URL string if successful, None otherwise
    """
    if not SUNRA_CLIENT_AVAILABLE:
        return None
        
    try:
        return sunra_client.upload(data, content_type, filename)
    except Exception as e:
        print(f"Warning: Failed to upload to Sunra: {e}")
        return None


def _resolve_file_path(file_path: str, subfolder: str) -> str:
    """
    Resolve file path by checking various locations.
    
    Args:
        file_path: Input file path
        subfolder: ComfyUI subfolder to check (e.g., "videos", "audio")
        
    Returns:
        Resolved absolute file path
        
    Raises:
        FileNotFoundError: If file cannot be found
    """
    # Check if it's already an absolute path that exists
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    
    # Try to find it in ComfyUI directories
    if FOLDER_PATHS_AVAILABLE:
        # Try specific subfolder
        resolved_path = folder_paths.get_full_path(subfolder, file_path)
        if resolved_path and os.path.exists(resolved_path):
            return resolved_path
        
        # Try general inputs folder
        if subfolder != "inputs":
            resolved_path = folder_paths.get_full_path("inputs", file_path)
            if resolved_path and os.path.exists(resolved_path):
                return resolved_path
    
    # Check if the path exists relative to current directory
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    
    raise FileNotFoundError(f"File not found: {file_path}")


def _download_file(url: str, output_path: Path, timeout: int = 60) -> None:
    """
    Download file from URL to specified path.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        timeout: Request timeout in seconds
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download from {url}: {str(e)}")


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
        # Return enum list as a normal dropdown
        config = {"default": schema_config.get("default", schema_config["enum"][0])}
        
        # Add tooltip for allow_custom_enum fields
        if schema_config.get("allow_custom_enum", False):
            config["tooltip"] = f"Select from list (or use {schema_config.get('name', 'field')}_override for custom values)"
        
        return (schema_config["enum"], config)

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
            return sunra_client.upload_image(value, image_format="png")
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


def file_to_base64(file_path: str, mime_type: str) -> str:
    """Convert file to base64 string with data URL."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_base64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{file_base64}"


def process_video_input(value: Any) -> Optional[str]:
    """
    Process video input for Sunra API.
    Converts video file path or VideoFromFile object to URL (preferred) or base64 string.
    """
    if value is None:
        return None

    # Handle VideoFromFile objects from ComfyUI
    if hasattr(value, '__class__') and value.__class__.__name__ == 'VideoFromFile':
        # VideoFromFile stores the file in a private __file attribute
        file_attr = getattr(value, '_VideoFromFile__file', None)
        
        if file_attr is None:
            raise ValueError(f"Cannot access file data from VideoFromFile object: {value}")
        
        if isinstance(file_attr, str):
            # It's a file path
            video_path = file_attr
        elif isinstance(file_attr, io.BytesIO):
            # It's BytesIO data
            file_attr.seek(0)
            video_data = file_attr.read()
            
            # Try to upload
            url = _upload_to_sunra(video_data, "video/mp4", "video.mp4")
            if url:
                return url
            
            # Fall back to base64
            video_base64 = base64.b64encode(video_data).decode("utf-8")
            return f"data:video/mp4;base64,{video_base64}"
        else:
            raise ValueError(f"Unexpected type for VideoFromFile.__file: {type(file_attr)}")
    elif isinstance(value, str):
        video_path = value
    else:
        raise ValueError(f"Unsupported video input type: {type(value)}")

    # Resolve and upload file path
    video_path = _resolve_file_path(video_path, "videos")
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(video_path)
    if mime_type is None:
        mime_type = "video/mp4"
    
    # Try to upload
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    url = _upload_to_sunra(video_data, mime_type, os.path.basename(video_path))
    if url:
        return url
    
    # Fallback to base64
    return file_to_base64(video_path, mime_type)


def process_audio_input(value: Any) -> Optional[str]:
    """
    Process audio input for Sunra API.
    Converts audio file path or ComfyUI AUDIO dict to URL (preferred) or base64 string.
    """
    if value is None:
        return None

    # Handle ComfyUI AUDIO dict format
    if isinstance(value, dict) and "waveform" in value:
        if "path" in value and value["path"]:
            # If we have a path, use that
            audio_path = value["path"]
        else:
            # Convert waveform tensor to audio file
            if TORCHAUDIO_AVAILABLE and value["waveform"] is not None:
                audio_path = None
                try:
                    waveform = value["waveform"]
                    sample_rate = value.get("sample_rate", 44100)
                    
                    # Create temporary audio file
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    audio_path = tmp_file.name
                    tmp_file.close()  # Close the file handle, we only need the path

                    torchaudio.save(audio_path, waveform.squeeze(0), sample_rate)
                    
                    # Read and upload
                    with open(audio_path, "rb") as f:
                        audio_data = f.read()
                    
                    url = _upload_to_sunra(audio_data, "audio/wav", "audio.wav")
                    
                    if url:
                        return url
                    
                    # Fallback to base64
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                    return f"data:audio/wav;base64,{audio_base64}"
                except Exception as e:
                    raise ValueError(f"Failed to process audio waveform: {e}")
                finally:
                    # Clean up temp file
                    if audio_path and os.path.exists(audio_path):
                        try:
                            os.unlink(audio_path)
                        except OSError as e:
                            print(f"Warning: Failed to delete temporary audio file {audio_path}: {e}")
            else:
                raise ValueError("Cannot process AUDIO dict without path or waveform data")
    elif isinstance(value, str):
        audio_path = value
    else:
        raise ValueError(f"Unsupported audio input type: {type(value)}")

    # Resolve and upload file path
    audio_path = _resolve_file_path(audio_path, "audio")
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = "audio/mpeg"
    
    # Try to upload
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    url = _upload_to_sunra(audio_data, mime_type, os.path.basename(audio_path))
    if url:
        return url
    
    # Fallback to base64
    return file_to_base64(audio_path, mime_type)


def process_video_output(data: Union[str, Dict]) -> Any:
    """
    Process video output from Sunra API.
    Downloads video and returns it as ComfyUI VideoFromFile.
    """
    from comfy_api.input_impl.video_types import VideoFromFile
    
    # Extract video URL from various formats
    video_url = data
    if isinstance(data, dict):
        video_url = data.get("url") or data.get("file_url") or str(data)
    
    if not isinstance(video_url, str) or not video_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid video URL: {video_url}")
    
    # Check cache
    if video_url in _media_cache:
        cached_data = _media_cache[video_url]
        return VideoFromFile(io.BytesIO(cached_data["data"]))
    
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
        _download_file(video_url, output_path)
    
    # Read video file into memory
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    
    # Cache the data
    _media_cache[video_url] = {"data": video_bytes, "path": str(output_path)}
    
    return VideoFromFile(io.BytesIO(video_bytes))


def _process_and_cache_audio(audio_url: str) -> Dict[str, Any]:
    """
    Helper to download, process, and cache audio data.
    """
    if audio_url in _media_cache:
        return _media_cache[audio_url]
    
    # Create output directory
    if FOLDER_PATHS_AVAILABLE:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(folder_paths.get_output_directory()) / "SunraAudio" / timestamp
    else:
        output_dir = Path(tempfile.gettempdir()) / "comfyui_sunra_audio"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL or generate one
    filename = audio_url.split("/")[-1].split("?")[0]
    if not filename.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")):
        filename = f"audio_{hash(audio_url)}.mp3"
    
    output_path = output_dir / filename
    
    # Download audio
    _download_file(audio_url, output_path)
    
    # Get relative path for ComfyUI
    if FOLDER_PATHS_AVAILABLE:
        relative_path = output_path.relative_to(Path(folder_paths.get_output_directory()))
    else:
        relative_path = output_path
    
    # Load audio
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
    _media_cache[audio_url] = result
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
    
    # Create output directory
    if FOLDER_PATHS_AVAILABLE:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(folder_paths.get_output_directory()) / "Sunra3D" / timestamp
    else:
        output_dir = Path(tempfile.gettempdir()) / "comfyui_sunra_3d"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = model_url.split("/")[-1].split("?")[0]
    
    # Check if it's an archive or direct model file
    is_archive = filename.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    
    if is_archive:
        # Download archive
        archive_path = output_dir / filename
        _download_file(model_url, archive_path, timeout=120)
        
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
            if FOLDER_PATHS_AVAILABLE:
                relative_path = model_files[0].relative_to(Path(folder_paths.get_output_directory()))
            else:
                relative_path = model_files[0]
            return str(relative_path)
        else:
            raise RuntimeError(f"No 3D model files found in archive from {model_url}")
    else:
        # Direct model file download
        if not filename.endswith((".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl")):
            filename = f"model_{hash(model_url)}.glb"
        
        output_path = output_dir / filename
        _download_file(model_url, output_path, timeout=120)
        
        # Return path relative to ComfyUI output folder
        if FOLDER_PATHS_AVAILABLE:
            relative_path = output_path.relative_to(Path(folder_paths.get_output_directory()))
        else:
            relative_path = output_path
        return str(relative_path)


def infer_type_from_name(name: str) -> Optional[str]:
    """
    Infer type from parameter name.
    Used as a fallback when type is not explicitly specified.
    """
    name_lower = name.lower()
    
    # Define type patterns
    type_patterns = {
        "image": ["image", "img", "photo", "picture"],
        "mask": ["mask"],
        "video": ["video", "movie", "clip"],
        "audio": ["audio", "sound", "music", "voice"],
        "integer": ["count", "number", "steps", "iterations"],
        "float": ["scale", "strength", "weight", "ratio"],
        "boolean": ["enable", "disable", "use", "is_", "has_", "should_"],
    }
    
    # Check patterns
    for type_name, patterns in type_patterns.items():
        if any(word in name_lower for word in patterns):
            return type_name
    
    # Default to string
    return "string"


def validate_value(value: Any, schema_type: str, schema_config: Dict[str, Any]) -> Any:
    """
    Validate and convert a value according to schema definition.
    """
    # Check enum values (skip for allow_custom_enum fields)
    if "enum" in schema_config and not schema_config.get("allow_custom_enum", False):
        if value not in schema_config["enum"]:
            raise ValueError(
                f"Value {value} not in allowed values: {schema_config['enum']}"
            )

    # Check min/max constraints
    if "min" in schema_config and value < schema_config["min"]:
        raise ValueError(f"Value {value} is below minimum {schema_config['min']}")

    if "max" in schema_config and value > schema_config["max"]:
        raise ValueError(f"Value {value} is above maximum {schema_config['max']}")

    # Type conversions
    type_conversions = {
        "integer": lambda v: int(v) if isinstance(v, float) else v,
        "float": lambda v: float(v) if isinstance(v, int) else v,
        "string": lambda v: str(v) if not isinstance(v, str) else v,
        "boolean": lambda v: v.lower() in ["true", "yes", "1", "on"] if isinstance(v, str) else bool(v),
    }
    
    if schema_type in type_conversions:
        return type_conversions[schema_type](value)
    
    return value