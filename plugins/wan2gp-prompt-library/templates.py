"""
Built-in starter templates for the prompt library
"""

from typing import Dict, List, Any


def get_starter_templates() -> Dict[str, List[Dict[str, Any]]]:
    """Get starter templates organized by collection

    Returns:
        Dict mapping collection_id to list of template prompts
    """
    return {
        "cinematic": [
            {
                "name": "Epic Drone Shot",
                "prompt": "Cinematic drone shot flying over {location}, golden hour lighting, volumetric fog, 4K quality, professional color grading, epic landscape",
                "negative_prompt": "blurry, distorted, low quality, pixelated, amateur",
                "tags": ["aerial", "landscape", "cinematic", "drone"],
                "variables": ["location"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            },
            {
                "name": "Slow Motion Action",
                "prompt": "Cinematic slow motion shot of {action}, dramatic lighting, high contrast, motion blur, film grain, professional cinematography",
                "negative_prompt": "static, boring, flat lighting, amateur",
                "tags": ["action", "slow-motion", "cinematic"],
                "variables": ["action"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Establishing Shot",
                "prompt": "Wide establishing shot of {location}, cinematic composition, dramatic sky, professional color grading, 4K quality, film look",
                "negative_prompt": "blurry, amateur, poor composition, oversaturated",
                "tags": ["wide-shot", "establishing", "cinematic", "landscape"],
                "variables": ["location"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            },
            {
                "name": "Close-up Portrait",
                "prompt": "Cinematic close-up portrait of {subject}, shallow depth of field, professional lighting, film grain, emotional expression, 4K quality",
                "negative_prompt": "blurry, distorted face, bad anatomy, oversaturated",
                "tags": ["portrait", "close-up", "cinematic", "character"],
                "variables": ["subject"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            }
        ],
        "anime": [
            {
                "name": "Anime Character Action",
                "prompt": "Anime style, {character} performing {action}, dynamic pose, vibrant colors, detailed animation, studio quality, clean lines",
                "negative_prompt": "blurry, poorly drawn, distorted anatomy, low quality",
                "tags": ["anime", "character", "action", "dynamic"],
                "variables": ["character", "action"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            },
            {
                "name": "Anime Background Scene",
                "prompt": "Anime background art, {location}, detailed scenery, soft lighting, pastel colors, studio ghibli style, atmospheric",
                "negative_prompt": "blurry, low detail, poor composition, distorted",
                "tags": ["anime", "background", "scenery", "ghibli"],
                "variables": ["location"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            },
            {
                "name": "Magical Girl Transformation",
                "prompt": "Magical girl transformation sequence, {character}, sparkles and ribbons, vibrant colors, dynamic camera movement, anime style, high quality animation",
                "negative_prompt": "static, boring, low quality, distorted",
                "tags": ["anime", "magical-girl", "transformation", "dynamic"],
                "variables": ["character"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            }
        ],
        "realistic": [
            {
                "name": "Nature Documentary",
                "prompt": "Nature documentary style, {subject} in natural habitat, photorealistic, professional wildlife cinematography, 4K quality, natural lighting",
                "negative_prompt": "artificial, staged, low quality, blurry, distorted",
                "tags": ["nature", "documentary", "wildlife", "realistic"],
                "variables": ["subject"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Urban Street Scene",
                "prompt": "Photorealistic urban street scene, {location}, natural lighting, people walking, modern city, 4K quality, professional cinematography",
                "negative_prompt": "artificial, cartoonish, low quality, distorted",
                "tags": ["urban", "street", "realistic", "city"],
                "variables": ["location"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Product Showcase",
                "prompt": "Professional product video, {product}, rotating 360 degrees, studio lighting, clean background, commercial quality, 4K resolution",
                "negative_prompt": "amateur, poor lighting, cluttered, low quality",
                "tags": ["product", "commercial", "studio", "realistic"],
                "variables": ["product"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            },
            {
                "name": "Time Lapse Scene",
                "prompt": "Time lapse of {scene}, photorealistic, smooth motion, dramatic lighting changes, professional quality, 4K resolution",
                "negative_prompt": "choppy, low quality, artificial, blurry",
                "tags": ["time-lapse", "realistic", "dramatic"],
                "variables": ["scene"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 30,
                    "guidance_scale": 7.5,
                }
            }
        ],
        "character": [
            {
                "name": "Character Walk Cycle",
                "prompt": "{character} walking towards camera, full body shot, smooth animation, consistent movement, professional quality, clear details",
                "negative_prompt": "jerky motion, distorted anatomy, inconsistent, blurry",
                "tags": ["character", "walk", "animation", "full-body"],
                "variables": ["character"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Character Talking",
                "prompt": "Close-up of {character} talking, natural lip sync, expressive face, subtle movements, professional animation quality",
                "negative_prompt": "frozen face, poor lip sync, distorted, unnatural",
                "tags": ["character", "talking", "close-up", "expression"],
                "variables": ["character"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Character Emotion",
                "prompt": "{character} expressing {emotion}, facial close-up, subtle movements, natural expression, professional quality, detailed animation",
                "negative_prompt": "frozen face, unnatural, distorted, poor animation",
                "tags": ["character", "emotion", "expression", "close-up"],
                "variables": ["character", "emotion"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            },
            {
                "name": "Character Action Sequence",
                "prompt": "{character} performing {action}, dynamic movement, smooth animation, action choreography, professional quality",
                "negative_prompt": "static, jerky, poor animation, distorted",
                "tags": ["character", "action", "dynamic", "animation"],
                "variables": ["character", "action"],
                "settings": {
                    "resolution": "1280x720",
                    "steps": 35,
                    "guidance_scale": 8.0,
                }
            }
        ]
    }


def initialize_library_with_templates(library):
    """Initialize a library with starter templates

    Args:
        library: PromptLibrary instance to populate
    """
    templates = get_starter_templates()

    for collection_id, prompts in templates.items():
        for prompt_data in prompts:
            library.add_prompt(
                collection_id=collection_id,
                name=prompt_data["name"],
                prompt=prompt_data["prompt"],
                negative_prompt=prompt_data.get("negative_prompt", ""),
                tags=prompt_data.get("tags", []),
                settings=prompt_data.get("settings", {})
            )
