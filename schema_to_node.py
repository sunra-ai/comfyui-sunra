"""
Schema to ComfyUI node converter.

This module handles parsing of schema definitions and converting them
into ComfyUI node specifications.
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict
from .type_converters import convert_schema_type_to_comfyui, infer_type_from_name


def parse_schema_inputs(schema: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Parse input definitions from schema and convert to ComfyUI INPUT_TYPES format.

    Args:
        schema: The complete schema dictionary

    Returns:
        Dictionary with "required" and "optional" keys containing input definitions
    """
    input_types = {"required": {}, "optional": {}}

    inputs = schema.get("inputs", {})

    # Process required inputs
    for name, config in inputs.get("required", {}).items():
        input_type, type_config = convert_schema_input(name, config)
        input_types["required"][name] = (input_type, type_config)

    # Process optional inputs
    for name, config in inputs.get("optional", {}).items():
        input_type, type_config = convert_schema_input(name, config)
        input_types["optional"][name] = (input_type, type_config)

    # Add force_rerun for caching control (like in replicate)
    input_types["optional"]["force_rerun"] = ("BOOLEAN", {"default": False})

    # Sort inputs by order if specified
    input_types = order_inputs_by_priority(input_types, inputs)

    return input_types


def convert_schema_input(name: str, config: Dict[str, Any]) -> Tuple[Any, Dict]:
    """
    Convert a single input definition to ComfyUI format.

    Args:
        name: Input parameter name
        config: Input configuration from schema

    Returns:
        Tuple of (type, config) for ComfyUI
    """
    # Get type from config or infer from name
    input_type = config.get("type")
    if not input_type:
        input_type = infer_type_from_name(name)

    # Add name to config for type converter
    config_with_name = config.copy()
    config_with_name["name"] = name

    return convert_schema_type_to_comfyui(input_type, config_with_name)


def order_inputs_by_priority(
    input_types: Dict[str, Dict], inputs_schema: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Order inputs based on the 'order' field in schema.

    Args:
        input_types: The parsed input types dictionary
        inputs_schema: The original inputs schema

    Returns:
        Ordered input types dictionary
    """

    def get_order(name: str, category: str) -> float:
        """Get order value for an input."""
        if category == "required":
            inputs = inputs_schema.get("required", {})
        else:
            inputs = inputs_schema.get("optional", {})

        return inputs.get(name, {}).get("order", float("inf"))

    # Create ordered dictionaries
    ordered_types = {"required": OrderedDict(), "optional": OrderedDict()}

    # Sort required inputs
    required_names = sorted(
        input_types["required"].keys(), key=lambda x: get_order(x, "required")
    )
    for name in required_names:
        ordered_types["required"][name] = input_types["required"][name]

    # Sort optional inputs (excluding force_rerun)
    optional_names = [n for n in input_types["optional"].keys() if n != "force_rerun"]
    optional_names.sort(key=lambda x: get_order(x, "optional"))

    for name in optional_names:
        ordered_types["optional"][name] = input_types["optional"][name]

    # Add force_rerun at the end
    if "force_rerun" in input_types["optional"]:
        ordered_types["optional"]["force_rerun"] = input_types["optional"][
            "force_rerun"
        ]

    return ordered_types


def parse_schema_outputs(
    schema: Dict[str, Any],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Parse output definitions from schema.

    Args:
        schema: The complete schema dictionary

    Returns:
        Tuple of (RETURN_TYPES, RETURN_NAMES)
    """
    outputs = schema.get("outputs", {})

    if not outputs:
        # Default to single string output
        return (("STRING",), ("output",))

    return_types = []
    return_names = []

    # Sort outputs by order if specified
    sorted_outputs = sorted(
        outputs.items(), key=lambda x: x[1].get("order", float("inf"))
    )

    for name, config in sorted_outputs:
        output_type = config.get("type", "string")

        # Map to ComfyUI types
        if output_type == "image":
            return_types.append("IMAGE")
        elif output_type == "video":
            return_types.append("VIDEO")
        elif output_type == "audio":
            return_types.append("AUDIO")
        elif output_type == "mask":
            return_types.append("MASK")
        elif output_type in ["integer", "int"]:
            return_types.append("INT")
        elif output_type == "float":
            return_types.append("FLOAT")
        elif output_type == "boolean":
            return_types.append("BOOLEAN")
        else:
            return_types.append("STRING")

        return_names.append(name)

    return (tuple(return_types), tuple(return_names))


def extract_node_metadata(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract node metadata from schema.

    Args:
        schema: The complete schema dictionary

    Returns:
        Dictionary with node metadata
    """
    metadata = {
        "name": schema.get("name", "Unnamed Node"),
        "display_name": schema.get("display_name", schema.get("name", "Unnamed Node")),
        "description": schema.get("description", ""),
        "category": schema.get("category", "Sunra.ai"),
        "version": schema.get("version", "1.0.0"),
    }

    # Add any additional metadata fields
    for key in ["author", "license", "documentation_url", "icon"]:
        if key in schema:
            metadata[key] = schema[key]

    return metadata


def validate_schema(schema: Dict[str, Any]) -> List[str]:
    """
    Validate a schema for required fields and consistency.

    Args:
        schema: The schema to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["model_id", "name", "api_endpoint"]
    for field in required_fields:
        if field not in schema:
            errors.append(f"Missing required field: {field}")

    # Check inputs structure
    if "inputs" in schema:
        if not isinstance(schema["inputs"], dict):
            errors.append("'inputs' must be a dictionary")
        else:
            # Check that inputs have required or optional sections
            if not schema["inputs"].get("required") and not schema["inputs"].get(
                "optional"
            ):
                errors.append(
                    "'inputs' must have either 'required' or 'optional' section"
                )

    # Check outputs structure
    if "outputs" in schema:
        if not isinstance(schema["outputs"], dict):
            errors.append("'outputs' must be a dictionary")
        else:
            # Check each output has a type
            for name, config in schema["outputs"].items():
                if not isinstance(config, dict) or "type" not in config:
                    errors.append(f"Output '{name}' must have a 'type' field")

    # Check API endpoint format
    if "api_endpoint" in schema:
        endpoint = schema["api_endpoint"]
        if not isinstance(endpoint, str) or not endpoint:
            errors.append("'api_endpoint' must be a non-empty string")

    return errors


def schema_to_comfyui_node_spec(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a complete schema to ComfyUI node specification.

    Args:
        schema: The complete schema dictionary

    Returns:
        Dictionary with all node specifications for ComfyUI
    """
    # Validate schema first
    errors = validate_schema(schema)
    if errors:
        raise ValueError(f"Invalid schema: {'; '.join(errors)}")

    # Parse components
    input_types = parse_schema_inputs(schema)
    return_types, return_names = parse_schema_outputs(schema)
    metadata = extract_node_metadata(schema)

    # Build node specification
    node_spec = {
        "input_types": input_types,
        "return_types": return_types,
        "return_names": return_names,
        "function": "execute",  # Standard function name
        "category": metadata["category"],
        "metadata": metadata,
        "api_endpoint": schema["api_endpoint"],
        "model_id": schema["model_id"],
    }

    # Add optional node attributes
    if schema.get("output_node", False):
        node_spec["output_node"] = True

    return node_spec


def load_schema_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load and validate a schema from a JSON file.

    Args:
        filepath: Path to the schema JSON file

    Returns:
        Validated schema dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validate the loaded schema
    errors = validate_schema(schema)
    if errors:
        raise ValueError(f"Invalid schema in {filepath}: {'; '.join(errors)}")

    return schema
