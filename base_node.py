"""
Base node class for Sunra ComfyUI nodes.

This module provides the base class that all dynamically generated
Sunra nodes inherit from.
"""

import time
import hashlib
import json
import io
from typing import Dict, Any, Tuple, List
import torch

from .type_converters import (
    process_image_input,
    process_image_output,
    process_video_input,
    process_video_output,
    process_audio_input,
    process_audio_output,
    process_audio_path_output,
    process_3d_output,
    validate_value,
)


class SunraBaseNode:
    """
    Base class for all Sunra ComfyUI nodes.

    This class handles common functionality like:
    - Input validation and processing
    - API communication
    - Output processing
    - Caching and change detection
    """

    def __init__(self, schema: Dict[str, Any], api_client):
        """
        Initialize the node with schema and API client.

        Args:
            schema: The node schema definition
            api_client: The Sunra API client instance
        """
        self.schema = schema
        self.api_client = api_client
        self.model_id = schema["model_id"]
        self.api_endpoint = schema["api_endpoint"]
        self.metadata = schema.get("metadata", {})

    @classmethod
    def create_from_schema(cls, schema: Dict[str, Any], api_client):
        """
        Create a dynamic node class from schema.

        Args:
            schema: The complete node schema
            api_client: The Sunra API client instance

        Returns:
            A new node class configured according to the schema
        """
        from .schema_to_node import schema_to_comfyui_node_spec

        # Parse schema to get node specifications
        node_spec = schema_to_comfyui_node_spec(schema)

        # Create the dynamic class
        class DynamicSunraNode(cls):
            # Set class attributes from schema
            CATEGORY = node_spec["category"]
            FUNCTION = node_spec["function"]
            OUTPUT_NODE = node_spec.get("output_node", False)

            # Store schema and api_client as class attributes
            _schema = schema
            _api_client = api_client

            def __init__(self):
                """Initialize without arguments for ComfyUI compatibility."""
                super().__init__(self._schema, self._api_client)

            @classmethod
            def INPUT_TYPES(cls):
                """Return input types for ComfyUI."""
                return node_spec["input_types"]

            RETURN_TYPES = node_spec["return_types"]
            RETURN_NAMES = node_spec["return_names"]

            @classmethod
            def IS_CHANGED(cls, **kwargs):
                """
                Determine if the node should be re-executed.

                Uses force_rerun flag and input hashing for caching.
                """
                if kwargs.get("force_rerun", False):
                    return time.time()

                # Create hash of all inputs
                input_hash = cls._create_input_hash(kwargs)
                return input_hash

            @classmethod
            def _create_input_hash(cls, inputs: Dict[str, Any]) -> str:
                """Create a hash of inputs for caching."""
                # Convert inputs to a hashable format
                hashable_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        # Hash tensor data
                        hashable_inputs[key] = hashlib.sha256(
                            value.cpu().numpy().tobytes()
                        ).hexdigest()[:16]
                    elif isinstance(value, (list, dict)):
                        # Convert to JSON string
                        hashable_inputs[key] = json.dumps(value, sort_keys=True)
                    else:
                        hashable_inputs[key] = str(value)

                # Create final hash
                content = json.dumps(hashable_inputs, sort_keys=True)
                return hashlib.sha256(content.encode()).hexdigest()[:16]

        # Return the class itself, not an instance
        return DynamicSunraNode

    def execute(self, **kwargs) -> Tuple:
        """
        Main execution function called by ComfyUI.

        Args:
            **kwargs: Input parameters from ComfyUI

        Returns:
            Tuple of outputs according to RETURN_TYPES
        """
        try:
            # Remove internal parameters
            kwargs = self._clean_inputs(kwargs)

            # Validate inputs
            validated_inputs = self._validate_inputs(kwargs)

            # Process inputs for API
            processed_inputs = self._process_inputs(validated_inputs)

            # Call Sunra API
            response = self._call_api(processed_inputs)

            # Process outputs
            outputs = self._process_outputs(response)

            # Return as tuple
            return tuple(outputs)

        except Exception as e:
            # Log error and re-raise
            print(f"Error in {self.model_id}: {str(e)}")
            raise

    def _clean_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal parameters like force_rerun."""
        cleaned = inputs.copy()
        cleaned.pop("force_rerun", None)
        return cleaned

    def _validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs according to schema."""
        validated = {}

        # Get input definitions from schema
        all_inputs = {}
        all_inputs.update(self.schema["inputs"].get("required", {}))
        all_inputs.update(self.schema["inputs"].get("optional", {}))

        for name, value in inputs.items():
            if name in all_inputs:
                config = all_inputs[name]
                validated[name] = validate_value(value, config["type"], config)
            else:
                # Pass through unknown inputs
                validated[name] = value

        return validated

    def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for API call."""
        processed = {}

        # Get all input definitions
        all_inputs = {
            **self.schema["inputs"].get("required", {}),
            **self.schema["inputs"].get("optional", {}),
        }

        for name, value in inputs.items():
            # Skip None values and special "None" strings
            if value is None or (isinstance(value, str) and value == "None"):
                continue

            input_config = all_inputs.get(name, {})
            input_type = input_config.get("type", "string")
            api_name = input_config.get("api_field_name", name)

            # Process based on type
            if input_type in ("image", "mask"):
                processed[api_name] = process_image_input(value)
            elif input_type == "video":
                processed[api_name] = process_video_input(value)
            elif input_type == "audio":
                processed[api_name] = process_audio_input(value)
            elif input_type == "boolean":
                processed[api_name] = bool(value)
            else:
                processed[api_name] = value

        return processed

    def _call_api(self, inputs: Dict[str, Any]) -> Any:
        """
        Call Sunra API with processed inputs.

        Args:
            inputs: Processed input parameters

        Returns:
            API response
        """
        # Use the API client to make the call
        return self.api_client.call(self.api_endpoint, inputs)

    def _process_outputs(self, response: Any) -> List[Any]:
        """
        Process API response to ComfyUI outputs.

        Args:
            response: Raw API response

        Returns:
            List of outputs matching RETURN_TYPES
        """
        output_schema = self.schema.get("outputs", {})
        return_names = self.__class__.RETURN_NAMES
        return_types = self.__class__.RETURN_TYPES

        outputs = []
        audio_data_cache = {}  # Cache to store audio data for path outputs
        
        for i, name in enumerate(return_names):
            # Check if this is an automatically added audio_path output
            if name.endswith("_path") and i > 0:
                audio_name = name[:-5]  # Remove '_path' suffix
                # Check if previous output was the corresponding audio
                if i > 0 and return_names[i-1] == audio_name and return_types[i-1] == "AUDIO":
                    # Use cached audio file path
                    if audio_name in audio_data_cache:
                        outputs.append(audio_data_cache[audio_name])
                    else:
                        outputs.append("")  # Default empty path
                    continue
            
            output_config = output_schema.get(name, {})
            output_type = output_config.get("type", "string")

            # Extract value from response
            if isinstance(response, dict):
                value = response.get(name, self._get_default_output(output_type))
            elif len(output_schema) == 1:  # Use original schema length, not return_names
                value = response  # Single output, response is the value
            else:
                value = self._get_default_output(output_type)

            # Process and convert to appropriate type
            if output_type == "audio" and name not in audio_data_cache:
                # Process audio and get path from global variable
                audio_dict = self._convert_output_value(value, output_type)
                audio_path = process_audio_path_output(value)
                outputs.append(audio_dict)
                audio_data_cache[name] = audio_path
            else:
                outputs.append(self._convert_output_value(value, output_type))

        return outputs

    def _convert_output_value(self, value: Any, output_type: str) -> Any:
        """Convert output value to the appropriate type."""
        if value is None:
            return self._get_default_output(output_type)

        type_converters = {
            "image": process_image_output,
            "mask": process_image_output,
            "video": process_video_output,
            "audio": process_audio_output,
            "3d": process_3d_output,
            "model": process_3d_output,  # Alternative name for 3D outputs
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
        }

        converter = type_converters.get(output_type)
        return converter(value) if converter else value

    def _get_default_output(self, output_type: str) -> Any:
        """Get default value for an output type."""
        if output_type == "video":
            from comfy_api.input_impl.video_types import VideoFromFile
            # Create empty video bytes
            empty_video = io.BytesIO()
            return VideoFromFile(empty_video)
        
        defaults = {
            "image": torch.zeros(1, 3, 512, 512),
            "mask": torch.zeros(1, 1, 512, 512),
            "string": "",
            "integer": 0,
            "float": 0.0,
            "boolean": False,
            "audio": {"waveform": torch.zeros(1, 1, 44100), "sample_rate": 44100},  # Empty 1-second mono audio
            "3d": "",     # Empty path for 3D models
            "model": "",  # Alternative name for 3D outputs
        }
        return defaults.get(output_type, None)
