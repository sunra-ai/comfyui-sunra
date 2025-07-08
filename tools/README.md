# Schema Generation Tools

Generate ComfyUI node schemas from Sunra AI OpenAPI definitions.

## Quick Start

```bash
# Generate all schemas
python generate_schemas.py

# Generate specific model
python generate_schemas.py --model flux-kontext-dev

# List available models
python generate_schemas.py --list

# Validate schemas
python generate_schemas.py --validate
```

## Adding New Models

1. Add to `sunra_models.json`:

```json
{
  "name": "your-model-name",
  "openapi_url": "https://openapi.sunra.ai/main/org/model/latest.json",
  "endpoint": "/org/model/endpoint",
  "overrides": {
    "model_id": "your-model-name",
    "name": "Display Name",
    "category": "Sunra.ai/Category"
  }
}
```

2. Generate: `python generate_schemas.py --model your-model-name`
3. Restart ComfyUI

## Structure

- `generate_schemas.py` - CLI entry point
- `sunra_models.json` - Model configurations
- `schema_generator/`
  - `converter.py` - OpenAPI → Schema conversion
  - `generator.py` - Generation logic
  - `models_config.py` - Loads models from JSON
  - `cache/` - Cached OpenAPI specs

## Type Detection

Auto-detects parameter types:
- **Images**: `image`, `input_image`, `mask`
- **Videos**: `video`, `input_video`
- **Audio**: `audio`, `voice`, `music`
- **Multiline**: `prompt`, `description`, `text`