# ComfyUI Sunra.ai Plugin Examples

This directory contains example workflows demonstrating the capabilities of the Sunra.ai plugin for ComfyUI.

## Available Examples

### 1. `flux_text_to_image.json`
**FLUX Kontext Text-to-Image Generation**

This workflow demonstrates basic text-to-image generation using FLUX Kontext models.

- **Model**: flux-context-pro
- **Features**: Text-to-image generation with configurable parameters
- **Output**: High-quality images based on text prompts

**How to use:**
1. Load this workflow in ComfyUI
2. Modify the prompt in the SunraFluxContext node
3. Adjust generation parameters as needed
4. Run the workflow

### 2. `flux_image_editing.json`
**FLUX Kontext Image Editing**

This workflow shows how to edit existing images using natural language instructions.

- **Model**: flux-context-pro
- **Features**: Context-aware image editing
- **Input**: Load your own image using the LoadImage node
- **Output**: Edited image based on your instructions

**How to use:**
1. Load this workflow in ComfyUI
2. Place your image in the ComfyUI input folder
3. Update the LoadImage node to use your image
4. Modify the edit prompt in the SunraImageEdit node
5. Run the workflow

### 3. `seedance_video.json`
**Seedance Video Generation**

This workflow demonstrates dance video generation using the Seedance model.

- **Model**: seedance-1.0-prp
- **Features**: Dance and motion synthesis
- **Output**: Video URL and preview frames

**How to use:**
1. Load this workflow in ComfyUI
2. Modify the dance prompt in the SunraSeedance node
3. Adjust video parameters (resolution, frame count, etc.)
4. Run the workflow
5. The video URL will be displayed in the ShowText node

## Prerequisites

Before running these examples, make sure you have:

1. **ComfyUI** installed and running
2. **Sunra.ai Plugin** installed in your ComfyUI custom_nodes directory
3. **SUNRA_KEY** environment variable set with your API key
4. **Dependencies** installed: `pip install -r requirements.txt`

## Loading Workflows

To load these workflows in ComfyUI:

1. Open ComfyUI in your browser
2. Click "Load" in the interface
3. Navigate to the `examples` folder
4. Select the workflow JSON file you want to use
5. The workflow will be loaded in the ComfyUI interface

## Customization Tips

### For FLUX Kontext Models:
- **Prompts**: Be specific and descriptive
- **Model Selection**: 
  - Use `flux-context-dev` for experimentation
  - Use `flux-context-pro` for production
  - Use `flux-context-max` for highest quality
- **Resolution**: 1024x1024 is recommended for best results
- **Steps**: 28 steps provide good quality/speed balance

### For Seedance:
- **Dance Styles**: Try "ballet", "hip-hop", "contemporary", "freestyle"
- **Motion Strength**: 0.7 is a good starting point
- **Frame Count**: 16-24 frames work well for short sequences
- **Reference Images**: Use clear, full-body images for best results

### For Image Editing:
- **Edit Prompts**: Be specific about what you want to change
- **Edit Strength**: Start with 0.7 and adjust as needed
- **Masks**: Use masks for precise control over edit areas

## Troubleshooting

**Common Issues:**

1. **API Key Error**: Make sure `SUNRA_KEY` is set in your environment
2. **Model Not Found**: Verify the model name is correct
3. **Workflow Won't Load**: Check that the plugin is installed correctly
4. **No Output**: Check the ComfyUI console for error messages

**Getting Help:**

- Check the main README.md for detailed installation instructions
- Visit the [Sunra.ai documentation](https://docs.sunra.ai)
- Report issues on the GitHub repository

## Contributing

Have an interesting workflow to share? We welcome contributions!

1. Create your workflow using the Sunra.ai nodes
2. Export it as a JSON file
3. Add it to this examples folder
4. Update this README with a description
5. Submit a pull request

---

**Note**: These examples require a valid Sunra.ai API key. Visit [sunra.ai](https://sunra.ai) to get started. 