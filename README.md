# Image-Background-Replacement-using-Stable-Diffusion

## Project Overview

This project demonstrates object replacement in images using Stable Diffusion 3, a cutting-edge generative AI model. The goal is to automate object replacement seamlessly by:

- Generating masks based on text prompts using CLIPSeg.
- Replacing objects using Stable Diffusion's inpainting capabilities.
- Enhancing results with blending techniques like Poisson blending.
  
## Key Features

1. Stable Diffusion 3 Integration
    - Leveraging the latest advancements for high-quality inpainting.
      
2. Automatic Mask Generation
    - Using CLIPSeg for text-prompt-based segmentation.
      
3. Seamless Object Replacement
    - Ensures photorealistic blending using improved Poisson blending.
      
4. Customizable Prompts
    - Replace any object in the image with just a text description.

## Setup

Step 1: Install Required Libraries

  ```
    pip install diffusers transformers torch accelerate huggingface_hub
  ```
Step 2: Hugging Face Login

Get your Hugging Face token from Hugging Face Settings and log in:

  ```
    from huggingface_hub import login
    login("Your_Hugging_Face_Token")  # Replace with your token
  ```

Step 3: Import Libraries

  ```
    import torch
    from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionInpaintPipeline
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
  ```

## How It Works

1. Model Loading
   
   Three models are used:
    
    - def load_base_model(): Stable Diffusion Base Model for generating images. 
    - def load_inpaint_model(): Stable Diffusion Inpainting Model for object replacement.
    - def load_segment_model(): CLIPSeg Segmentation Model for automatic mask generation.

2. Mask Generation
   
    def generate_mask(image, text_prompt, processor, model): Generates binary mask using CLIPSeg model.

3. Object Replacement
   
    def replace_object(image_path, object_prompt, replacement_prompt, processor, seg_model, inpaint_pipe): Performs object 
    replacement using the inpainting model and mask.

4. Result Visualization
   
    def display_results(original, mask, result): Displays results side by side using Matplotlib.

## Example Usage

1. Save your input image (e.g., Deer.jpg) to the working directory.
   
2. Update the parameters in the main function:
   
  - image_path: Path to the input image.
  - object_prompt: Description of the object to be replaced.
  - replacement_prompt: Description of the replacement object.

3. Run the Program

## Full Pipeline

1. Input Image
   Provide an image containing the object to be replaced.

2. Mask Generation
   Generate a mask isolating the object described in the text prompt.

3. Object Replacement
   Replace the object with a new one described in the replacement prompt.

4. Blending and Post-Processing
   Use Poisson blending for seamless integration.

5. Result Visualization
   Display side-by-side results for evaluation.

## Acknowledgments

   - Stable Diffusion by Stability AI
   - CLIPSeg by CIDAS for segmentation
   - Hugging Face for model hosting and tools
  
## License

This project is released under the MIT License.

## Feel free to contribute and enhance this project! Happy coding! 😊
