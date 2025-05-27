# VACE ControlNet Guide

VACE is a powerful ControlNet that enables Video-to-Video and Reference-to-Video generation. It allows you to inject your own images into output videos, animate characters, perform inpainting/outpainting, and continue videos.

## Overview

VACE is probably one of the most powerful Wan models available. With it, you can:
- Inject people or objects into scenes
- Animate characters
- Perform video inpainting and outpainting
- Continue existing videos
- Transfer motion from one video to another
- Change the style of scenes while preserving depth

## Getting Started

### Model Selection
1. Select either "Vace 1.3B" or "Vace 13B" from the dropdown menu
2. Note: VACE works best with videos up to 7 seconds with the Riflex option enabled

### Input Types

VACE accepts three types of visual hints (which can be combined):

#### 1. Control Video
- Transfer motion or depth to a new video
- Use only the first n frames and extrapolate the rest
- Perform inpainting with grey color (127) as mask areas
- Grey areas will be filled based on text prompt and reference images

#### 2. Reference Images
- Use as background/setting for the video
- Inject people or objects of your choice
- Select multiple reference images
- **Tip**: Replace complex backgrounds with white for better object integration
- Always describe injected objects/people explicitly in your text prompt

#### 3. Video Mask
- Stronger control over which parts to keep (black) or replace (white)
- Perfect for inpainting/outpainting
- Example: White mask except at beginning/end (black) keeps first/last frames while generating middle content

## Common Use Cases

### Motion Transfer
**Goal**: Animate a character of your choice using motion from another video
**Setup**: 
- Reference Images: Your character
- Control Video: Person performing desired motion
- Text Prompt: Describe your character and the action

### Object/Person Injection
**Goal**: Insert people or objects into a scene
**Setup**:
- Reference Images: The people/objects to inject
- Text Prompt: Describe the scene and explicitly mention the injected elements

### Character Animation
**Goal**: Animate a character based on text description
**Setup**:
- Control Video: Video of person moving
- Text Prompt: Detailed description of your character

### Style Transfer with Depth
**Goal**: Change scene style while preserving spatial relationships
**Setup**:
- Control Video: Original video (for depth information)
- Text Prompt: New style description

## Integrated Matanyone Tool

WanGP includes the Matanyone tool, specifically tuned for VACE workflows. This helps create control videos and masks simultaneously.

### Creating Face Replacement Masks
1. Load your video in Matanyone
2. Click on the face in the first frame
3. Create a mask for the face
4. Generate both control video and mask video with "Generate Video Matting"
5. Export to VACE with "Export to current Video Input and Video Mask"
6. Load replacement face image in Reference Images field

### Advanced Matanyone Tips
- **Negative Point Prompts**: Remove parts from current selection
- **Sub Masks**: Create multiple independent masks, then combine them
- **Background Masks**: Select everything except the character (useful for background replacement)
- Enable/disable sub masks in Matanyone settings

## Recommended Settings

### Quality Settings
- **Skip Layer Guidance**: Turn ON with default configuration for better results
- **Long Prompts**: Use detailed descriptions, especially for background elements not in reference images
- **Steps**: Use at least 15 steps for good quality, 30+ for best results

### Sliding Window Settings
For very long videos, configure sliding windows properly:

- **Window Size**: Set appropriate duration for your content
- **Overlap Frames**: Long enough for motion continuity, short enough to avoid blur propagation
- **Discard Last Frames**: Remove at least 4 frames from each window (VACE 1.3B tends to blur final frames)

### Background Removal
VACE includes automatic background removal options:
- Use for reference images containing people/objects
- **Don't use** for landscape/setting reference images (first reference image)
- Multiple background removal types available

## Window Sliding for Long Videos

Generate videos up to 1 minute by merging multiple windows:

### How It Works
- Each window uses corresponding time segment from control video
- Example: 0-4s control video → first window, 4-8s → second window, etc.
- Automatic overlap management ensures smooth transitions

### Settings
- **Window Size**: Duration of each generation window
- **Overlap Frames**: Frames shared between windows for continuity
- **Discard Last Frames**: Remove poor-quality ending frames
- **Add Overlapped Noise**: Reduce quality degradation over time

### Formula
```
Generated Frames = [Windows - 1] × [Window Size - Overlap - Discard] + Window Size
```

### Multi-Line Prompts (Experimental)
- Each line of prompt used for different window
- If more windows than prompt lines, last line repeats
- Separate lines with carriage return

## Advanced Features

### Extend Video
Click "Extend the Video Sample, Please!" during generation to add more windows dynamically.

### Noise Addition
Add noise to overlapped frames to hide accumulated errors and quality degradation.

### Frame Truncation
Automatically remove lower-quality final frames from each window (recommended: 4 frames for VACE 1.3B).

## External Resources

### Official VACE Resources
- **GitHub**: https://github.com/ali-vilab/VACE/tree/main/vace/gradios
- **User Guide**: https://github.com/ali-vilab/VACE/blob/main/UserGuide.md
- **Preprocessors**: Gradio tools for preparing materials

### Recommended External Tools
- **Annotation Tools**: For creating precise masks
- **Video Editors**: For preparing control videos
- **Background Removal**: For cleaning reference images

## Troubleshooting

### Poor Quality Results
1. Use longer, more detailed prompts
2. Enable Skip Layer Guidance
3. Increase number of steps (30+)
4. Check reference image quality
5. Ensure proper mask creation

### Inconsistent Windows
1. Increase overlap frames
2. Use consistent prompting across windows
3. Add noise to overlapped frames
4. Reduce discard frames if losing too much content

### Memory Issues
1. Use VACE 1.3B instead of 13B
2. Reduce video length or resolution
3. Decrease window size
4. Enable quantization

### Blurry Results
1. Reduce overlap frames
2. Increase discard last frames
3. Use higher resolution reference images
4. Check control video quality

## Tips for Best Results

1. **Detailed Prompts**: Describe everything in the scene, especially elements not in reference images
2. **Quality Reference Images**: Use high-resolution, well-lit reference images
3. **Proper Masking**: Take time to create precise masks with Matanyone
4. **Iterative Approach**: Start with short videos, then extend successful results
5. **Background Preparation**: Remove complex backgrounds from object/person reference images
6. **Consistent Lighting**: Match lighting between reference images and intended scene 