# VACE ControlNet Guide

VACE is a powerful ControlNet that enables Video-to-Video and Reference-to-Video generation. It allows you to inject your own images into output videos, animate characters, perform inpainting/outpainting, and continue existing videos.

## Overview

VACE is probably one of the most powerful Wan models available. With it, you can:
- Inject people or objects into scenes
- Animate characters
- Perform video inpainting and outpainting
- Continue existing videos
- Transfer motion from one video to another
- Change the style of scenes while preserving the structure of the scenes


## Getting Started

### Model Selection
1. Select either "Vace 1.3B" or "Vace 13B" from the dropdown menu
2. Note: VACE works best with videos up to 7 seconds with the Riflex option enabled

You can also use any derived Vace models such as Vace Fusionix or combine Vace with Loras accelerator such as Causvid.

### Input Types

#### 1. Control Video
The Control Video is the source material that contains the instructions about what you want. So Vace expects in the Control Video some visual hints about the type of processing expected: for instance replacing an area by something else, converting an Open Pose wireframe into a human motion, colorizing an Area,  transferring the depth of an image area, ...

For example, anywhere your control video contains the color 127 (grey), it will be considered as an area to be inpainting and replaced by the content of your text prompt and / or a reference image (see below). Likewise if the frames of a Control Video contains an Open Pose wireframe (basically some straight lines tied together that describes the pose of a person), Vace will automatically turn this Open Pose into a real human based on the text prompt and any reference Images (see below).

You can either build yourself the Control Video with the annotators tools provided by the Vace team (see the Vace ressources at the bottom) or you can let WanGP (recommended option) generates on the fly a Vace formatted Control Video based on information you provide.

WanGP wil need the following information to generate a Vace Control Video:
- A *Control Video* : this video shouldn't have been altered by an annotator tool and can be taken straight from youtube or your camera
- *Control Video Process* : This is the type of process you want to apply on the control video. For instance *Transfer Human Motion* will generate the Open Pose information from your video so that you can transfer this same motion to a generated character. If you want to do only *Spatial Outpainting* or *Temporal Inpainting / Outpainting* you may want to choose the *Keep Unchanged* process.
- *Area Processed* : you can target the processing to a specific area. For instance even if there are multiple people in the Control Video you may want to replace only one them. If you decide to target an area you will need to provide a *Video Mask* as well. These types of videos can be easily created using the Matanyone tool embedded with WanGP (see the doc of Matanyone below). WanGP can apply different types of process, one the mask and another one on the outside the mask.

Another nice thing is that you can combine all effects above with Outpainting since WanGP will create automatically an outpainting area in the Control Video if you ask for this. 

By default WanGP will ask Vace to generate new frames in the "same spirit" of the control video if the latter is shorter than the number frames that you have requested.

Be aware that the Control Video and Video Mask will be before anything happens resampled to the number of frames per second of Vace (usually 16) and resized to the output size you have requested.
#### 2. Reference Images
With Reference Images you can inject people or objects of your choice in the Video.
You can also force Images to appear at a specific frame nos in the Video.

If the Reference Image is a person or an object, it is recommended to turn on the background remover that will replace the background by the white color.
This is not needed for a background image or an injected frame at a specific position.

It is recommended to describe injected objects/people explicitly in your text prompt so that Vace can connect the Reference Images to the new generated video and this will increase the chance that you will find your injected people or objects.


### Understanding Vace Control Video and Mask format
As stated above WanGP will adapt the Control Video and the Video Mask to meet your instructions. You can preview the first frames of the new Control Video and of the Video Mask in the Generation Preview box (just click a thumbnail) to check that your request has been properly interpreted. You can as well ask WanGP to save in the main folder of WanGP the full generated Control Video and  Video Mask by launching the app with the *--save-masks* command.

Look at the background colors of both the Control Video and the Video Mask:
The Mask Video is the most important because depending on the color of its pixels, the Control Video will be interpreted differently. If an area in the Mask is black, the corresponding Control Video area will be kept as is. On the contrary if an area of the Mask is plain white, a Vace process will be applied on this area. If there isn't any Mask Video the Vace process will apply on the whole video frames. The nature of the process itself will depend on what there is in the Control Video for this area. 
- if the area is grey (127) in the Control Video, this area will be replaced by new content based on the text prompt or image references
- if an area represents a person in the wireframe Open Pose format, it will be replaced by a person animated with motion described by the Open Pose.The appearance of the person will depend on the text prompt or image references
- if an area contains multiples shades of grey, these will be assumed to represent different levels of image depth and Vace will try to generate new content located at the same depth

There are more Vace representations. For all the different mapping please refer the official Vace documentation.

### Other Processing
Most of the processing below and the ones related to Control Video can be combined together.
- **Temporal Outpainting**\
Temporal Outpainting requires an existing *Source Video* or *Control Video* and it amounts to adding missing frames. It is implicit if you use a Source Video that you want to continue (new frames will be added at the end of this Video) or if you provide a Control Video that contains fewer frames than the number that you have requested to generate.

- **Temporal Inpainting**\
With temporal inpainting you are asking Vace to generate missing frames that should exist between existing frames. There are two ways to do that:
    - *Injected Reference Images* : Each Image is injected a position of your choice and Vace will fill the gaps between these frames
    - *Frames to keep in Control Video* : If using a Control Video, you can ask WanGP to hide some of these frames to let Vace generate "alternate frames" for these parts of the Control Video.

- **Spatial Outpainting**\
This feature creates new content to the top, bottom, left or right of existing frames of a Control Video. You can set the amount of content for each direction by specifying a percentage of extra content in relation to the existing frame. Please note that the resulting video will target the resolution you specified. So if this Resolution corresponds to that of your Control Video you may lose details. Therefore it may be relevant to pick a higher resolution with Spatial Outpainting.\
There are two ways to do Spatial Outpainting:
    - *Injected Reference Frames* : new content will be added around Injected Frames
    - *Control Video* : new content will be added on all the frames of the whole Control Video


### Example 1 : Replace a Person in one video by another one by keeping the Background
1) In Vace, select *Control Video Process*=**Transfer human pose**, *Area processed*=**Masked area** 
2) In *Matanyone Video Mask Creator*, load your source video and create a mask where you targetted a specific person 
3) Click *Export to Control Video Input and Video Mask Input* to transfer both the original video that now becomes the *Control Video* and the black & white mask that now defines the *Video Mask Area*
4) Back in Vace, in *Reference Image* select **Inject Landscapes / People / Objects** and upload one or several pictures of the new person
5) Generate

This works also with several people at the same time (you just need to mask several people in *Matanyone*), you can also play with the slider *Expand / Shrink Mask* if the new person is larger than the original one and of course, you can also use the text *Prompt* if you dont want to use an image for the swap.


### Example 2 : Change the Background behind some characters
1) In Vace, select *Control Video Process*=**Inpainting**, *Area processed*=**Non Masked area** 
2) In *Matanyone Video Mask Creator*, load your source video and create a mask where you targetted the people you want to keep 
3) Click *Export to Control Video Input and Video Mask Input* to transfer both the original video that now becomes the *Control Video* and the black & white mask that now defines the *Video Mask Area*
4) Generate

If instead *Control Video Process*=**Depth**, then the background although it will be still different it will have a similar geometry than in the control video

### Example 3 : Outpaint a Video to the Left and Inject a Character in this new area 
1) In Vace, select *Control Video Process*=**Keep Unchanged** 
2) *Control Video Outpainting in Percentage* enter the value 40 to the *Left* entry
3) In *Reference Image* select **Inject Landscapes / People / Objects** and upload one or several pictures of a person
4) Enter the *Prompt* such as "a person is coming from the left" (you will need of course a more accurate description)
5) Generate



### Creating Face / Object Replacement Masks
Matanyone is a tool that will generate the Video Mask that needs to be combined with the Control Video. It is very useful as you just need to indicate in the first frame the area you want to mask and it will compute masked areas for the following frames by taking into account the motion.
1. Load your video in Matanyone
2. Click on the face or object in the first frame
3. Validate the mask by clicking **Set Mask**
4. Generate a copy of the control video (for easy transfers) and a new mask video by clicking "Generate Video Matting"
5. Export to VACE with *Export to Control Video Input and Video Mask Input*

### Advanced Matanyone Tips
- **Negative Point Prompts**: Remove parts from current selection if the mask goes beyond the desired area
- **Sub Masks**: Create multiple independent masks, then combine them. This may be useful if you are struggling to select exactly what you want.    



## Window Sliding for Long Videos
Generate videos up to 1 minute by merging multiple windows:
The longer the video the greater the quality degradation. However the effect will be less visible if your generated video reuses mostly non altered control video.

When this feature is enabled it is important to keep in mind that every positional argument of Vace (frames positions of *Injected Reference Frames*, *Frames to keep in Control Video*) are related to the first frame of the first Window. This is convenient as changing the size of a sliding window won't have any impact and this allows you define in advance the inject frames for all the windows.

Likewise, if you use *Continue Video File* by providing a *Source Video*, this Source Video will be considered as the first window and the positional arguments will be calculated in relation to the first frame of this Source Video. Also the *overlap window size* parameter will correspond to the number of frames used of the Source Video that is temporally outpainted to produce new content.

### How It Works
- Each window uses the corresponding time segment of the Control Video
- Example: 0-4s control video → first window, 4-8s → second window, etc.
- Automatic overlap management ensures smooth transitions


### Formula
This formula gives the number of Generated Frames for a specific number of Sliding Windows :
```
Generated Frames = [Nb Windows - 1] × [Window Size - Overlap - Discard] + Window Size
```

### Multi-Line Prompts (Experimental)
If you enable *Text Prompts separated by a Carriage Return will be used for a new Sliding Window*, you can define in advance a different prompt for each window.:
- Each prompt is separated by a Carriage Return  
- Each line of prompt will be used for a different window
- If more windows than prompt lines, last line repeats

## Recommended Settings

### Quality Settings
- **Skip Layer Guidance**: Turn ON with default configuration for better results (useless with FusioniX of Causvid are there is no cfg)
- **Long Prompts**: Use detailed descriptions, especially for background elements not in reference images
- **Steps**: Use at least 15 steps for good quality, 30+ for best results if you use the original Vace model. But only 8-10 steps are sufficient with Vace Funsionix or if you use Loras such as Causvid or Self-Forcing.

### Sliding Window Settings
For very long videos, configure sliding windows properly:

- **Window Size**: Set appropriate duration for your content
- **Overlap Frames**: Long enough for motion continuity, short enough to avoid blur propagation
- **Discard Last Frames**: Remove at least 4 frames from each window (VACE 1.3B tends to blur final frames)
- **Add Overlapped Noise**: May or may not reduce quality degradation over time

### Background Removal
WanGP includes automatic background removal options:
- Use for reference images containing people/objects
- **Don't use** this for landscape/setting reference images (the first reference image)
- If you are not happy with the automatic background removal tool you can use the Image version of Matanyone for a precise background removal

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