#!/usr/bin/env python3
"""
Convert OpenAPI schema to ComfyUI node schema format.

This module handles the conversion of OpenAPI definitions from Sunra API
to the schema format used by the dynamic node system.
"""

import json
import requests
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from urllib.parse import urlparse
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import OPENAPI_TYPE_MAP, IMAGE_PARAMS, VIDEO_PARAMS, AUDIO_PARAMS, MULTILINE_PATTERNS


class OpenAPIToSchemaConverter:
    """Convert OpenAPI schemas to ComfyUI node schemas."""
    
    # Use constants from the main module
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_openapi_schema(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch OpenAPI schema from URL with caching."""
        # Create cache filename from URL
        url_path = urlparse(url).path.replace("/", "_")
        # Remove .json suffix if already present
        if url_path.endswith(".json"):
            url_path = url_path[:-5]
        cache_file = self.cache_dir / (url_path + ".json")
        
        if use_cache and cache_file.exists():
            print(f"Using cached schema: {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        print(f"Fetching schema from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        schema = response.json()
        
        # Cache the response
        with open(cache_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        return schema
    
    def extract_model_info(self, openapi_schema: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Extract model information from OpenAPI schema."""
        # Extract from endpoint path
        parts = endpoint.strip("/").split("/")
        
        # Default values
        model_info = {
            "model_id": "-".join(parts),
            "name": " ".join(p.title() for p in parts),
            "api_endpoint": endpoint.strip("/"),
        }
        
        # Try to get info from OpenAPI metadata
        if "info" in openapi_schema:
            info = openapi_schema["info"]
            if "title" in info:
                model_info["display_name"] = info["title"]
            if "description" in info:
                model_info["description"] = info["description"]
        
        # Extract category from path
        if len(parts) >= 2:
            model_info["category"] = f"Sunra.ai/{parts[0].replace('-', ' ').title()}"
        
        return model_info
    
    def get_parameter_type(self, param_name: str, param_schema: Dict[str, Any]) -> str:
        """Determine parameter type from schema and name, prioritizing schema definitions."""
        
        openapi_type = param_schema.get("type")
        param_format = param_schema.get("format")

        # 1. Handle explicit non-string primitive types first.
        if openapi_type in ["integer", "number", "boolean"]:
            return OPENAPI_TYPE_MAP[openapi_type]

        # 2. Handle strings, which could be regular strings, enums, or file URIs.
        if openapi_type == "string":
            # If it's a URI, it's a media file. Use hints to determine which kind.
            if param_format == "uri":
                description = param_schema.get("description", "").lower()
                param_lower = param_name.lower()
                if any(img in param_lower for img in IMAGE_PARAMS) or 'image' in description:
                    return "image"
                if any(vid in param_lower for vid in VIDEO_PARAMS) or 'video' in description:
                    return "video"
                if any(aud in param_lower for aud in AUDIO_PARAMS) or 'audio' in description:
                    return "audio"
            
            # If it's not a URI, it's just a string (or an enum, handled later).
            return "string"
            
        # 3. Fallback for undefined or object types - use name hints.
        param_lower = param_name.lower()
        if any(img in param_lower for img in IMAGE_PARAMS):
            return "image"
        if any(vid in param_lower for vid in VIDEO_PARAMS):
            return "video"
        if any(aud in param_lower for aud in AUDIO_PARAMS):
            return "audio"
            
        # Default to string if we can't figure it out.
        return "string"
    
    def convert_parameter(self, name: str, param_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI parameter to our schema format."""
        param_type = self.get_parameter_type(name, param_schema)
        
        result = {
            "type": param_type,
            "description": param_schema.get("description", f"{name} parameter"),
        }
        
        # Add enum values
        if "enum" in param_schema:
            result["enum"] = param_schema["enum"]
        
        # Add default value
        if "default" in param_schema:
            result["default"] = param_schema["default"]
        
        # Add constraints for numeric types
        if param_type in ["integer", "float"]:
            if "minimum" in param_schema:
                result["min"] = param_schema["minimum"]
            if "maximum" in param_schema:
                result["max"] = param_schema["maximum"]
            
            # Add step
            if param_type == "integer":
                result["step"] = int(param_schema.get("multipleOf", 1))
            else:
                result["step"] = float(param_schema.get("multipleOf", 0.1))
        
        # Handle multiline strings (prompts)
        if param_type == "string" and any(word in name.lower() for word in MULTILINE_PATTERNS):
            result["multiline"] = True
        
        # Add order from x-sr-order if available
        if "x-sr-order" in param_schema:
            result["order"] = param_schema["x-sr-order"]
        
        return result
    
    def extract_inputs(self, openapi_schema: Dict[str, Any], endpoint: str, 
                      allow_custom_enum_fields: List[str] = None) -> Dict[str, Any]:
        """Extract input parameters from OpenAPI schema."""
        inputs = {"required": {}, "optional": {}}
        if allow_custom_enum_fields is None:
            allow_custom_enum_fields = []
        
        # Find the endpoint definition
        paths = openapi_schema.get("paths", {})
        endpoint_def = None
        
        for path, methods in paths.items():
            if path == endpoint:
                endpoint_def = methods.get("post", methods.get("get"))
                break
        
        if not endpoint_def:
            print(f"Warning: Endpoint {endpoint} not found in OpenAPI schema")
            return inputs
        
        # Get request body schema
        request_body = endpoint_def.get("requestBody", {})
        content = request_body.get("content", {})
        json_content = content.get("application/json", {})
        schema_ref = json_content.get("schema", {})
        
        # Resolve schema reference
        if "$ref" in schema_ref:
            schema_name = schema_ref["$ref"].split("/")[-1]
            input_schema = openapi_schema["components"]["schemas"].get(schema_name, {})
        else:
            input_schema = schema_ref
        
        # Get required fields
        required_fields = input_schema.get("required", [])
        
        # Convert properties
        properties = input_schema.get("properties", {})
        
        # Sort by x-sr-order if available
        sorted_props = sorted(
            properties.items(),
            key=lambda x: x[1].get("x-sr-order", 999)
        )
        
        for idx, (prop_name, prop_schema) in enumerate(sorted_props):
            converted = self.convert_parameter(prop_name, prop_schema)
            
            # Check if this field should be allow_custom_enum
            if prop_name in allow_custom_enum_fields and "enum" in converted:
                converted["allow_custom_enum"] = True
            
            # Set order if not already set
            if "order" not in converted:
                converted["order"] = idx
            
            # Determine if required or optional
            if prop_name in required_fields:
                inputs["required"][prop_name] = converted
            else:
                inputs["optional"][prop_name] = converted
        
        return inputs
    
    def extract_outputs(self, openapi_schema: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Extract output definitions from OpenAPI schema."""
        outputs = {}
        
        # For Sunra APIs, the actual output is defined in the /requests/{request_id} endpoint
        paths = openapi_schema.get("paths", {})
        request_endpoint = paths.get("/requests/{request_id}")
        
        if request_endpoint and "get" in request_endpoint:
            # Get the 200 response schema
            responses = request_endpoint["get"].get("responses", {})
            success_response = responses.get("200", {})
            content = success_response.get("content", {})
            json_content = content.get("application/json", {})
            schema_ref = json_content.get("schema", {})
            
            # Resolve the schema reference
            if "$ref" in schema_ref:
                schema_name = schema_ref["$ref"].split("/")[-1]
                schemas = openapi_schema.get("components", {}).get("schemas", {})
                
                if schema_name in schemas:
                    output_schema = schemas[schema_name]
                    properties = output_schema.get("properties", {})
                    
                    for prop_name, prop_def in properties.items():
                        # Skip metadata fields
                        if prop_name in ["output_video_tokens", "request_id", "status"]:
                            continue
                        
                        # Check if it's an array of items
                        is_array = prop_def.get("type") == "array"
                        if is_array and "items" in prop_def:
                            item_def = prop_def["items"]
                        else:
                            item_def = prop_def
                        
                        # Determine output type
                        if "$ref" in item_def and "SunraFile" in item_def["$ref"]:
                            # It's a file output (or array of files)
                            # Check for 3D models first (before image) to handle cases like "ImageTo3DOutput"
                            if ("model" in prop_name.lower() or 
                                "3d" in schema_name.lower() or 
                                "to3d" in schema_name.lower() or
                                "3d" in endpoint.lower()):
                                # Handle 3D model outputs
                                outputs[prop_name] = {
                                    "type": "model",
                                    "description": prop_def.get("description", "Generated 3D model")
                                }
                            elif "video" in prop_name.lower() or "video" in schema_name.lower():
                                outputs[prop_name] = {
                                    "type": "video",
                                    "description": prop_def.get("description", "Generated videos" if is_array else "Generated video")
                                }
                            elif "audio" in prop_name.lower() or "audio" in schema_name.lower():
                                outputs[prop_name] = {
                                    "type": "audio",
                                    "description": prop_def.get("description", "Generated audio")
                                }
                            elif "image" in prop_name.lower() or "image" in schema_name.lower():
                                outputs[prop_name] = {
                                    "type": "image",
                                    "description": prop_def.get("description", "Generated images" if is_array else "Generated image")
                                }
                        else:
                            # Regular output
                            outputs[prop_name] = {
                                "type": OPENAPI_TYPE_MAP.get(prop_def.get("type", "string"), "string"),
                                "description": prop_def.get("description", "Output")
                            }
        
        # Default output if none found
        if not outputs:
            outputs["result"] = {
                "type": "string",
                "description": "API result"
            }
        
        return outputs
    
    def generate_schema(self, openapi_url: str, endpoint: str, 
                       custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate ComfyUI node schema from OpenAPI."""
        # Fetch OpenAPI schema
        openapi_schema = self.fetch_openapi_schema(openapi_url)
        
        # Extract model info
        model_info = self.extract_model_info(openapi_schema, endpoint)
        
        # Get allow_custom_enum_fields from overrides
        allow_custom_enum_fields = []
        if custom_overrides and "allow_custom_enum_fields" in custom_overrides:
            allow_custom_enum_fields = custom_overrides["allow_custom_enum_fields"]
        
        # Extract inputs and outputs
        inputs = self.extract_inputs(openapi_schema, endpoint, allow_custom_enum_fields)
        outputs = self.extract_outputs(openapi_schema, endpoint)
        
        # Build final schema
        schema = {
            **model_info,
            "version": "1.0.0",
            "inputs": inputs,
            "outputs": outputs,
        }
        
        # Apply custom overrides if provided
        if custom_overrides:
            schema.update(custom_overrides)
        
        return schema
    
    def save_schema(self, schema: Dict[str, Any], output_path: str):
        """Save schema to file."""
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Schema saved to: {output_path}")