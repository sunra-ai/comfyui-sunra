# ComfyUI Sunra.ai Plugin

A professional ComfyUI plugin for integrating Sunra.ai's cutting-edge AI models, including FLUX.1 Kontext and Seedance, with enhanced UI features and robust error handling.

## ✨ Features

### 🎨 **FLUX.1 Kontext Models**
State-of-the-art text-to-image and image-to-image generation with context-aware capabilities:
- **`flux-context-dev`** - Development version for research and experimentation
- **`flux-context-pro`** - Production-ready model with balanced speed and quality  
- **`flux-context-max`** - Maximum quality model with enhanced prompt adherence

### 💃 **Seedance 1.0 PRP**
Advanced dance and motion video generation with customizable styles and parameters

### 🎛️ **Advanced Image Editing** 
Context-aware image modifications using FLUX Kontext models with optional masking support

### 📊 **Queue Management**
Real-time monitoring of long-running requests with progress tracking

### 🚀 **Enhanced UI Features**
- Real-time progress bars and status indicators
- Smart caching with `IS_CHANGED` optimization
- Professional error handling and validation
- Client-side notifications and feedback
- Tooltips and helpful UI enhancements

## Installation

### Prerequisites

1. **ComfyUI**: Make sure you have ComfyUI installed and running
2. **Sunra.ai API Key**: Get your API key from [sunra.ai](https://sunra.ai)

### Set your Sunra.ai API key

Make sure you set your `SUNRA_KEY` environment variable before running ComfyUI:

#### On macOS or Linux:
```bash
export SUNRA_KEY="your-api-key-here"
python main.py
```

#### On Windows:
```cmd
set SUNRA_KEY="your-api-key-here"
python main.py
```

### Direct installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/comfyui-sunra
cd comfyui-sunra
pip install -r requirements.txt
```

## Available Nodes

### 1. Sunra.ai FLUX Kontext

Advanced image generation and editing using FLUX.1 Kontext models.

**Features:**
- Text-to-image generation
- Image-to-image transformation
- Multiple aspect ratios and sizes
- Advanced prompt adherence
- Safety checking

**Model Options:**
- `flux-context-dev` - 12B parameter model for research (fastest)
- `flux-context-pro` - Balanced performance for production use
- `flux-context-max` - Maximum quality and prompt adherence

**Parameters:**
- **Prompt**: Text description for image generation
- **Model**: Choose from dev, pro, or max variants
- **Size**: Resolution options (512x512 to 1344x768)
- **Inference Steps**: Number of denoising steps (1-100)
- **Guidance Scale**: Prompt adherence strength (0.0-20.0)
- **Seed**: Reproducibility control
- **Output Format**: JPEG, PNG, or WebP
- **Input Image**: Optional for image-to-image workflows

### 2. Sunra.ai Seedance

Generate dynamic dance and motion videos using the Seedance 1.0 PRP model.

**Features:**
- Text-to-dance video generation
- Reference image-based motion synthesis
- Customizable dance styles and music tempo
- Frame rate and duration control

**Parameters:**
- **Prompt**: Description of the dance or motion
- **Dimensions**: Width and height (256-1024px)
- **Animation**: Frame count, FPS, and duration
- **Motion Strength**: Intensity of movement (0.0-1.0)
- **Dance Style**: Freestyle, ballet, hip-hop, etc.
- **Music Tempo**: Slow, medium, or fast
- **Reference Image**: Optional starting pose

### 3. Sunra.ai Image Edit

Advanced image editing with context-aware modifications.

**Features:**
- Natural language edit instructions
- Mask-based selective editing
- Preserve original content option
- Professional-grade results

**Parameters:**
- **Image**: Input image to edit
- **Edit Prompt**: Description of desired changes
- **Model**: FLUX Kontext variant to use
- **Edit Strength**: Intensity of modifications (0.0-1.0)
- **Mask**: Optional mask for selective editing
- **Preserve Original**: Maintain original image structure

### 4. Sunra.ai Queue Status

Monitor the status of long-running requests.

**Features:**
- Real-time status updates
- Progress tracking
- Error handling

## Usage Examples

### Basic Text-to-Image Generation

1. Add the "Sunra.ai FLUX Kontext" node to your workflow
2. Set your prompt: "a majestic mountain landscape at sunset"
3. Choose model: "flux-context-pro"
4. Set desired size: "1024x1024"
5. Run the workflow

### Image Editing

1. Load an image into the "Sunra.ai Image Edit" node
2. Set edit prompt: "change the sky to a starry night"
3. Adjust edit strength: 0.7
4. Run to see the edited result

### Dance Video Generation

1. Add the "Sunra.ai Seedance" node
2. Set prompt: "a person performing contemporary dance"
3. Configure video parameters (frames, FPS, duration)
4. Optionally add a reference image
5. Generate the dance video

## Model Specifications

### FLUX.1 Kontext Models

| Model | Parameters | Use Case | Speed | Quality |
|-------|------------|----------|-------|---------|
| flux-context-dev | 12B | Research, experimentation | Fastest | Good |
| flux-context-pro | - | Production workflows | Balanced | High |
| flux-context-max | - | Maximum quality needs | Slower | Highest |

### Seedance 1.0 PRP

- **Type**: Video generation model
- **Specialization**: Dance and motion synthesis
- **Output**: Video files with preview frames
- **Supported Formats**: MP4, WebM

## Best Practices

### For FLUX Kontext Models:

1. **Prompt Engineering**: Be specific and detailed in your descriptions
2. **Model Selection**: 
   - Use `dev` for quick iterations and experiments
   - Use `pro` for production workflows
   - Use `max` when you need the highest quality
3. **Resolution**: Start with 1024x1024 for best results
4. **Inference Steps**: 28 steps provide good balance of quality and speed

### For Seedance:

1. **Reference Images**: Use clear, full-body images for best results
2. **Motion Strength**: Start with 0.7 and adjust based on desired intensity
3. **Frame Count**: 16-24 frames work well for short dance sequences
4. **Dance Style**: Be specific about the style you want

### For Image Editing:

1. **Edit Prompts**: Use clear, specific instructions
2. **Edit Strength**: Start with 0.7 and adjust as needed
3. **Masks**: Use masks for precise control over edit areas
4. **Preserve Original**: Enable when you want to maintain the original structure

## API Integration

This plugin uses the `sunra-client` library to communicate with Sunra.ai's API. The client handles:

- Authentication with API keys
- Request/response formatting
- Error handling and retries
- Queue management for long-running tasks

## Troubleshooting

### Common Issues:

1. **API Key Not Set**: Make sure `SUNRA_KEY` environment variable is set
2. **Model Not Found**: Verify the model name is correct
3. **Image Format Issues**: Ensure images are in supported formats (PNG, JPEG, WebP)
4. **Memory Issues**: For large images, consider reducing resolution or batch size

### Error Messages:

- `SUNRA_KEY environment variable is required`: Set your API key
- `No images returned from Sunra.ai API`: Check your prompt and model selection
- `Error generating image with FLUX Kontext`: Review your parameters and try again

## Contributing

We welcome contributions! Please feel free to:

- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the MIT License.

## Support

- **Documentation**: [docs.sunra.ai](https://docs.sunra.ai)
- **Community**: Join our Discord server
- **Issues**: Report problems on GitHub
- **API Support**: Contact Sunra.ai support

## Changelog

### v1.0.0
- Initial release with FLUX Kontext models
- Seedance 1.0 PRP support
- Advanced image editing capabilities
- Queue status monitoring

---

**Note**: This plugin requires a valid Sunra.ai API key. Visit [sunra.ai](https://sunra.ai) to get started. 