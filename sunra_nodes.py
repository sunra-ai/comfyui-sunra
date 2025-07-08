"""
Sunra.ai Custom Nodes for ComfyUI

This module provides custom nodes for integrating Sunra.ai models with ComfyUI,
including FLUX.1 Kontext models for image generation/editing and Seedance for video generation.
"""

import os
import io
import base64
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torch
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass

# Import dependencies
try:
    import sunra_client
except ImportError:
    raise ImportError(
        "sunra-client is required but not installed. "
        "Please install with: pip install sunra-client"
    )

try:
    import requests
except ImportError:
    raise ImportError(
        "requests is required but not installed. "
        "Please install with: pip install requests"
    )


# Utility functions
def validate_api_key() -> str:
    """Validate and return the Sunra.ai API key."""
    api_key = os.getenv('SUNRA_KEY')
    if not api_key:
        raise ValueError(
            "SUNRA_KEY environment variable is required. "
            "Get your API key from https://sunra.ai"
        )
    return api_key


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to PIL Image."""
    # Handle batch dimension
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
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
            return Image.fromarray(image_np, 'RGB')
        elif image_np.shape[2] == 4:  # RGBA
            return Image.fromarray(image_np, 'RGBA')
        elif image_np.shape[2] == 1:  # Grayscale
            return Image.fromarray(image_np.squeeze(2), 'L')
    elif len(image_np.shape) == 2:  # Grayscale
        return Image.fromarray(image_np, 'L')
    
    raise ValueError(f"Unsupported tensor shape: {image_np.shape}")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Add batch dimension
    image_tensor = torch.from_numpy(image_np[None, :, :, :])
    
    return image_tensor


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format.upper())
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


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


def create_request_hash(*args) -> str:
    """Create a hash for request parameters to help with caching."""
    # Convert all arguments to strings and create hash
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_request_id_from_response(response: Any) -> Optional[str]:
    """Extract x-request-id from response object or headers."""
    request_id = None
    try:
        # Try to get request_id from response object directly
        if hasattr(response, 'headers') and response.headers:
            request_id = response.headers.get('x-request-id')
        elif isinstance(response, dict):
            # Try to get from response dict
            if 'headers' in response:
                request_id = response['headers'].get('x-request-id')
            elif 'request_id' in response:
                request_id = response['request_id']
            elif 'x-request-id' in response:
                request_id = response['x-request-id']
    except Exception:
        # If we can't extract request_id, continue without it
        pass
    
    return request_id


def format_error_with_request_id(error_msg: str, request_id: Optional[str] = None) -> str:
    """Format error message with request ID if available."""
    if request_id:
        return f"{error_msg} (Request ID: {request_id})"
    return error_msg



class SunraFluxKontextDevNode:
    """
    FLUX Kontext Dev is a 12B image editing model that enables precise text- and image-guided edits while preserving character consistency.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "a beautiful landscape, highly detailed, cinematic lighting"
                }),
                "model": ([
                    "flux-kontext-dev", 
                ], {"default": "flux-kontext-dev"}),
                "aspect_ratio": ([
                    "None",
                    "1:1",
                    "2:3",
                    "3:2",
                    "3:4",
                    "4:3",
                    "16:9",
                    "9:16",
                    "21:9",
                    "9:21"
                ], {"default": "None"}),
                "number_of_steps": ("INT", {
                    "default": 28, "min": 1, "max": 100, "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 0.0, "min": 0.0, "max": 2147483647.0, "step": 1
                }),
                "output_format": (["jpg", "png", "webp"], {"default": "jpg"}),
                "output_quality": ("INT", {
                    "default": 80, "min": 1, "max": 100, "step": 1
                }),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enable_base64_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "Sunra.ai/FLUX Kontext"
    
    @classmethod
    def IS_CHANGED(cls, prompt: str, model: str, aspect_ratio: str, number_of_steps: int,
                   guidance_scale: float, seed: int, output_format: str,
                   output_quality: int, input_image: Optional[torch.Tensor] = None,
                   enable_safety_checker: bool = True, enable_base64_output: bool = False) -> str:
        """
        Determine if the node should be re-executed based on input changes.
        Returns a hash of the inputs for caching optimization.
        """
        # Create hash of all parameters that affect the output
        input_hash = None
        if input_image is not None:
            # Create a hash of the input image
            image_np = input_image.cpu().numpy()
            input_hash = hashlib.sha256(image_np.tobytes()).hexdigest()[:16]
        
        return create_request_hash(
            prompt, model, aspect_ratio, number_of_steps, guidance_scale,
            seed, output_format, output_quality, input_hash,
            enable_safety_checker, enable_base64_output
        )

    def generate_image(self, prompt: str, model: str, aspect_ratio: str, number_of_steps: int, 
                      guidance_scale: float, seed: int, output_format: str, 
                      output_quality: int, input_image: Optional[torch.Tensor] = None,
                      enable_safety_checker: bool = True, enable_base64_output: bool = False) -> Tuple[torch.Tensor]:
        """
        Generate images using FLUX.1 Kontext models.
        
        Args:
            prompt: Text description for image generation
            model: FLUX Kontext model variant to use
            aspect_ratio: Output image aspect ratio
            number_of_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            seed: Random seed for reproducibility
            output_format: Image format (jpeg, png, webp)
            output_quality: Compression quality for lossy formats
            input_image: Optional input image for image-to-image generation
            enable_safety_checker: Whether to check content safety
            enable_base64_output: Whether to return base64 encoded images
            
        Returns:
            Tuple containing generated images as tensor
        """
        response = None
        try:
            # Validate API key
            validate_api_key()
            
            # Prepare arguments for API call
            arguments = {
                "prompt": prompt,
                "num_inference_steps": number_of_steps,
                "guidance_scale": guidance_scale,
                "output_format": output_format,
                "output_quality": output_quality,
                "enable_safety_checker": enable_safety_checker,
                "enable_base64_output": enable_base64_output,
            }
            
            # Add aspect ratio if specified
            if aspect_ratio != "None":
                arguments["aspect_ratio"] = aspect_ratio
            
            # Add seed if specified
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle input image for image-to-image generation
            if input_image is not None:
                pil_image = tensor_to_pil(input_image)
                # Upload image using sunra_client
                image_url = sunra_client.upload_image(pil_image, format="png")
                arguments["image"] = image_url
                
            # Make API call
            response = sunra_client.subscribe(
                f"black-forest-labs/{model}/image-to-image",
                arguments=arguments,
                with_logs=True,
                on_enqueue=print,
                on_queue_update=print,
            )
            
            print(response)
            # Process response
            image = None
            if "image" in response and response["image"]:
                img_data = response["image"]
                try:
                    if "url" in img_data:
                        # Download image from URL
                        image = download_image_from_url(img_data["url"])
                    elif "base64" in img_data:
                        # Decode base64 image
                        img_bytes = base64.b64decode(img_data["base64"])
                        image = Image.open(io.BytesIO(img_bytes))
                    else:
                        print(f"Warning: Skipping image data without URL or base64: {img_data}")
                        
                except Exception as img_error:
                    print(f"Warning: Failed to process image: {str(img_error)}")
            
            if not image:
                raise ValueError("No valid image returned from Sunra.ai API")
            
            # Convert to tensor and return single image
            image_tensor = pil_to_tensor(image)
            return (image_tensor,)
            
        except ValueError as ve:
            # Re-raise validation errors as-is
            raise ve
        except Exception as e:
            # Wrap other errors with context
            request_id = extract_request_id_from_response(response)
            raise RuntimeError(format_error_with_request_id(f"FLUX Kontext generation failed: {str(e)}", request_id))



class SunraFluxContextNode:
    """
    FLUX.1 Kontext Node for ComfyUI
    
    Advanced image generation and editing using Sunra.ai's FLUX.1 Kontext models.
    Supports text-to-image and image-to-image generation with context-aware capabilities.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "a beautiful landscape, highly detailed, cinematic lighting"
                }),
                "model": ([
                    "flux-context-dev", 
                    "flux-context-pro", 
                    "flux-context-max"
                ], {"default": "flux-context-pro"}),
                "size": ([
                    "512x512", "768x768", "1024x1024", 
                    "1152x896", "896x1152", "1344x768", "768x1344"
                ], {"default": "1024x1024"}),
                "num_inference_steps": ("INT", {
                    "default": 28, "min": 1, "max": 100, "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "num_images": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1
                }),
                "output_format": (["jpeg", "png", "webp"], {"default": "jpeg"}),
                "output_quality": ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1
                }),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "enable_base64_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "Sunra.ai/FLUX Kontext"
    
    @classmethod
    def IS_CHANGED(cls, prompt: str, model: str, size: str, num_inference_steps: int,
                   guidance_scale: float, num_images: int, seed: int, output_format: str,
                   output_quality: int, input_image: Optional[torch.Tensor] = None,
                   enable_safety_checker: bool = True, enable_base64_output: bool = False) -> str:
        """
        Determine if the node should be re-executed based on input changes.
        Returns a hash of the inputs for caching optimization.
        """
        # Create hash of all parameters that affect the output
        input_hash = None
        if input_image is not None:
            # Create a hash of the input image
            image_np = input_image.cpu().numpy()
            input_hash = hashlib.sha256(image_np.tobytes()).hexdigest()[:16]
        
        return create_request_hash(
            prompt, model, size, num_inference_steps, guidance_scale,
            num_images, seed, output_format, output_quality, input_hash,
            enable_safety_checker, enable_base64_output
        )

    def generate_image(self, prompt: str, model: str, size: str, num_inference_steps: int, 
                      guidance_scale: float, num_images: int, seed: int, output_format: str, 
                      output_quality: int, input_image: Optional[torch.Tensor] = None,
                      enable_safety_checker: bool = True, enable_base64_output: bool = False) -> Tuple[torch.Tensor]:
        """
        Generate images using FLUX.1 Kontext models.
        
        Args:
            prompt: Text description for image generation
            model: FLUX Kontext model variant to use
            size: Output image dimensions
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            output_format: Image format (jpeg, png, webp)
            output_quality: Compression quality for lossy formats
            input_image: Optional input image for image-to-image generation
            enable_safety_checker: Whether to check content safety
            enable_base64_output: Whether to return base64 encoded images
            
        Returns:
            Tuple containing generated images as tensor
        """
        response = None
        try:
            # Validate API key
            validate_api_key()
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Prepare arguments for API call
            arguments = {
                "prompt": prompt,
                "size": size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "output_format": output_format,
                "output_quality": output_quality,
                "enable_safety_checker": enable_safety_checker,
                "enable_base64_output": enable_base64_output,
            }
            
            # Add seed if specified
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle input image for image-to-image generation
            if input_image is not None:
                pil_image = tensor_to_pil(input_image)
                # Upload image using sunra_client
                image_url = sunra_client.upload_image(pil_image, format="png")
                arguments["image"] = image_url
                
            # Make API call
            response = sunra_client.subscribe(
                f"black-forest-labs/{model}/image-to-image",
                arguments=arguments,
                with_logs=True,
                on_enqueue=print,
                on_queue_update=print,
            )
            
            # Process response
            images = []
            if "images" in response and response["images"]:
                for img_data in response["images"]:
                    try:
                        if "url" in img_data:
                            # Download image from URL
                            image = download_image_from_url(img_data["url"])
                        elif "base64" in img_data:
                            # Decode base64 image
                            img_bytes = base64.b64decode(img_data["base64"])
                            image = Image.open(io.BytesIO(img_bytes))
                        else:
                            print(f"Warning: Skipping image data without URL or base64: {img_data}")
                            continue
                        
                        # Convert to tensor and add to results
                        image_tensor = pil_to_tensor(image)
                        images.append(image_tensor)
                        
                    except Exception as img_error:
                        print(f"Warning: Failed to process image: {str(img_error)}")
                        continue
            
            if not images:
                raise ValueError("No valid images returned from Sunra.ai API")
            
            # Concatenate all images
            result = torch.cat(images, dim=0)
            return (result,)
            
        except ValueError as ve:
            # Re-raise validation errors as-is
            raise ve
        except Exception as e:
            # Wrap other errors with context
            request_id = extract_request_id_from_response(response)
            raise RuntimeError(format_error_with_request_id(f"FLUX Kontext generation failed: {str(e)}", request_id))

class SunraSeedanceNode:
    """
    Seedance Video Generation Node for ComfyUI
    
    Generate dance and motion videos using Sunra.ai's Seedance 1.0 PRP model.
    Supports text-to-dance generation with customizable styles and motion parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "a person performing an elegant contemporary dance"
                }),
                "model": (["seedance-1.0-prp"], {"default": "seedance-1.0-prp"}),
                "width": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64
                }),
                "height": ("INT", {
                    "default": 512, "min": 256, "max": 1024, "step": 64
                }),
                "num_frames": ("INT", {
                    "default": 16, "min": 8, "max": 64, "step": 1
                }),
                "fps": ("INT", {
                    "default": 24, "min": 8, "max": 60, "step": 1
                }),
                "duration": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1
                }),
                "motion_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "dance_style": ("STRING", {"default": "freestyle"}),
                "music_tempo": (["slow", "medium", "fast"], {"default": "medium"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_url", "preview_frames")
    FUNCTION = "generate_dance"
    CATEGORY = "Sunra.ai/Seedance"
    
    @classmethod
    def IS_CHANGED(cls, prompt: str, model: str, width: int, height: int,
                   num_frames: int, fps: int, duration: float, motion_strength: float,
                   seed: int, reference_image: Optional[torch.Tensor] = None,
                   dance_style: str = "freestyle", music_tempo: str = "medium") -> str:
        """
        Determine if the node should be re-executed based on input changes.
        """
        ref_hash = None
        if reference_image is not None:
            image_np = reference_image.cpu().numpy()
            ref_hash = hashlib.sha256(image_np.tobytes()).hexdigest()[:16]
        
        return create_request_hash(
            prompt, model, width, height, num_frames, fps, duration,
            motion_strength, seed, ref_hash, dance_style, music_tempo
        )

    def generate_dance(self, prompt: str, model: str, width: int, height: int, 
                      num_frames: int, fps: int, duration: float, motion_strength: float,
                      seed: int, reference_image: Optional[torch.Tensor] = None,
                      dance_style: str = "freestyle", music_tempo: str = "medium") -> Tuple[str, torch.Tensor]:
        """
        Generate dance videos using Seedance model.
        
        Args:
            prompt: Description of the dance or motion
            model: Seedance model to use
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames to generate
            fps: Frames per second
            duration: Video duration in seconds
            motion_strength: Intensity of movement (0.0-1.0)
            seed: Random seed for reproducibility
            reference_image: Optional starting pose image
            dance_style: Style of dance (e.g., 'ballet', 'hip-hop', 'contemporary')
            music_tempo: Music tempo for movement timing
            
        Returns:
            Tuple containing video URL and preview frames tensor
        """
        response = None
        try:
            # Validate API key
            validate_api_key()
            
            # Prepare arguments for API call
            arguments = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "duration": duration,
                "motion_strength": motion_strength,
                "dance_style": dance_style,
                "music_tempo": music_tempo,
            }
            
            # Add seed if specified
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle reference image
            if reference_image is not None:
                pil_image = tensor_to_pil(reference_image)
                # Upload image using sunra_client
                image_url = sunra_client.upload_image(pil_image, format="png")
                arguments["reference_image"] = image_url
                
            # Make API call
            response = sunra_client.subscribe(model, arguments=arguments)
            
            # Handle response
            video_url = ""
            preview_frames = None
            
            if "video" in response:
                video_url = response["video"]["url"]
            elif "video_url" in response:
                video_url = response["video_url"]
            else:
                print("Warning: No video URL found in response")
            
            # Extract preview frames if available
            if "preview_frames" in response and response["preview_frames"]:
                frames = []
                for frame_data in response["preview_frames"]:
                    try:
                        if "url" in frame_data:
                            image = download_image_from_url(frame_data["url"])
                        elif "base64" in frame_data:
                            img_bytes = base64.b64decode(frame_data["base64"])
                            image = Image.open(io.BytesIO(img_bytes))
                        else:
                            continue
                        
                        # Convert to tensor and add to frames
                        frame_tensor = pil_to_tensor(image)
                        frames.append(frame_tensor)
                        
                    except Exception as frame_error:
                        print(f"Warning: Failed to process preview frame: {str(frame_error)}")
                        continue
                
                if frames:
                    preview_frames = torch.cat(frames, dim=0)
            
            # Create placeholder frames if none were provided
            if preview_frames is None:
                placeholder = torch.zeros(1, height, width, 3, dtype=torch.float32)
                preview_frames = placeholder
            
            return (video_url, preview_frames)
            
        except ValueError as ve:
            # Re-raise validation errors as-is
            raise ve
        except Exception as e:
            # Wrap other errors with context
            request_id = extract_request_id_from_response(response)
            raise RuntimeError(format_error_with_request_id(f"Seedance video generation failed: {str(e)}", request_id))

class SunraImageEditNode:
    """
    FLUX.1 Kontext Image Editing Node for ComfyUI
    
    Advanced image editing using Sunra.ai's FLUX.1 Kontext models.
    Supports context-aware image modifications with optional masking.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "change the background to a beautiful sunset scene"
                }),
                "model": ([
                    "flux-context-dev", 
                    "flux-context-pro", 
                    "flux-context-max"
                ], {"default": "flux-context-pro"}),
                "edit_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1
                }),
                "num_inference_steps": ("INT", {
                    "default": 28, "min": 1, "max": 100, "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "preserve_original": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "Sunra.ai/FLUX Kontext"
    
    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, edit_prompt: str, model: str,
                   edit_strength: float, num_inference_steps: int, guidance_scale: float,
                   seed: int, mask: Optional[torch.Tensor] = None,
                   preserve_original: bool = True) -> str:
        """
        Determine if the node should be re-executed based on input changes.
        """
        # Create hash of the input image
        image_np = image.cpu().numpy()
        image_hash = hashlib.sha256(image_np.tobytes()).hexdigest()[:16]
        
        # Create hash of the mask if provided
        mask_hash = None
        if mask is not None:
            mask_np = mask.cpu().numpy()
            mask_hash = hashlib.sha256(mask_np.tobytes()).hexdigest()[:16]
        
        return create_request_hash(
            image_hash, edit_prompt, model, edit_strength,
            num_inference_steps, guidance_scale, seed,
            mask_hash, preserve_original
        )

    def edit_image(self, image: torch.Tensor, edit_prompt: str, model: str, 
                  edit_strength: float, num_inference_steps: int, guidance_scale: float,
                  seed: int, mask: Optional[torch.Tensor] = None, 
                  preserve_original: bool = True) -> Tuple[torch.Tensor]:
        """
        Edit images using FLUX.1 Kontext models.
        
        Args:
            image: Input image tensor to edit
            edit_prompt: Description of the desired changes
            model: FLUX Kontext model variant to use
            edit_strength: Strength of the editing operation (0.0-1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            seed: Random seed for reproducibility
            mask: Optional mask for selective editing
            preserve_original: Whether to preserve non-edited areas
            
        Returns:
            Tuple containing the edited image tensor
        """
        response = None
        try:
            # Validate API key
            validate_api_key()
            
            # Upload image using sunra_client
            pil_image = tensor_to_pil(image)
            image_url = sunra_client.upload_image(pil_image, format="png")
            
            # Prepare arguments for API call
            arguments = {
                "prompt": edit_prompt,
                "image": image_url,
                "edit_strength": edit_strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "preserve_original": preserve_original,
            }
            
            # Add seed if specified
            if seed != -1:
                arguments["seed"] = seed
            
            # Handle mask if provided
            if mask is not None:
                # Convert mask tensor to PIL Image
                mask_np = mask.squeeze().cpu().numpy()
                if mask_np.max() <= 1.0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                
                mask_pil = Image.fromarray(mask_np, mode='L')
                # Upload mask using sunra_client
                mask_url = sunra_client.upload_image(mask_pil, format="png")
                arguments["mask"] = mask_url
                
            # Make API call
            response = sunra_client.subscribe(model, arguments=arguments)
            
            # Process response
            if "images" in response and response["images"]:
                img_data = response["images"][0]
                
                try:
                    if "url" in img_data:
                        result_image = download_image_from_url(img_data["url"])
                    elif "base64" in img_data:
                        img_bytes = base64.b64decode(img_data["base64"])
                        result_image = Image.open(io.BytesIO(img_bytes))
                    else:
                        raise ValueError("No valid image data in response")
                    
                    # Convert to tensor
                    image_tensor = pil_to_tensor(result_image)
                    return (image_tensor,)
                    
                except Exception as img_error:
                    raise RuntimeError(f"Failed to process edited image: {str(img_error)}")
            else:
                raise ValueError("No images returned from Sunra.ai API")
                
        except ValueError as ve:
            # Re-raise validation errors as-is
            raise ve
        except Exception as e:
            # Wrap other errors with context
            request_id = extract_request_id_from_response(response)
            raise RuntimeError(format_error_with_request_id(f"FLUX Kontext image editing failed: {str(e)}", request_id))

class SunraQueueStatusNode:
    """
    Queue Status Monitor Node for ComfyUI
    
    Monitor the status and progress of long-running Sunra.ai requests.
    Useful for tracking video generation and complex image processing tasks.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "request_id": ("STRING", {
                    "default": "", 
                    "placeholder": "Enter request ID from Sunra.ai"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("status", "result", "progress")
    FUNCTION = "check_status"
    CATEGORY = "Sunra.ai/Utilities"
    OUTPUT_NODE = True  # This node produces output that should always be executed

    def check_status(self, request_id: str) -> Tuple[str, str, float]:
        """
        Check the status of a Sunra.ai request.
        
        Args:
            request_id: The unique identifier for the Sunra.ai request
            
        Returns:
            Tuple containing (status, result, progress)
            - status: Current status of the request ('pending', 'processing', 'completed', 'failed')
            - result: Result data or error message
            - progress: Progress percentage (0.0 to 1.0)
        """
        if not request_id or not request_id.strip():
            return ("error", "No request ID provided", 0.0)
        
        status_response = None
        try:
            # Validate API key
            validate_api_key()
            
            # Check status using sunra_client
            status_response = sunra_client.status(request_id.strip())
            
            status = status_response.get("status", "unknown")
            result = status_response.get("result", "")
            progress = float(status_response.get("progress", 0.0))
            
            # Ensure progress is in valid range
            progress = max(0.0, min(1.0, progress))
            
            return (status, result, progress)
            
        except ValueError as ve:
            # API key validation error
            return ("error", str(ve), 0.0)
        except Exception as e:
            # Other errors (network, parsing, etc.)
            req_id = extract_request_id_from_response(status_response)
            return ("error", format_error_with_request_id(f"Failed to check status: {str(e)}", req_id), 0.0)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SunraFluxContext": SunraFluxContextNode,
    "SunraFluxKontextDevNode": SunraFluxKontextDevNode,
    "SunraSeedance": SunraSeedanceNode,
    "SunraImageEdit": SunraImageEditNode,
    "SunraQueueStatus": SunraQueueStatusNode,
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "SunraFluxContext": "FLUX.1 Kontext Generator",
    "SunraFluxKontextDevNode": "FLUX.1 Kontext Dev",
    "SunraSeedance": "Seedance Video Generator",
    "SunraImageEdit": "FLUX.1 Kontext Image Editor", 
    "SunraQueueStatus": "Sunra Queue Monitor",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 