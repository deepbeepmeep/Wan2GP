# FINETUNES

A Finetuned model is model that shares the same architecture of one specific model but has derived weights from this model. Some finetuned models have been created by combining multiple finetuned models.

As there are potentially an infinite number of finetunes, specific finetuned models are not known by default by WanGP. However you can create a finetuned model definition that will tell WanGP about the existence of this finetuned model and WanGP will do as usual all the work for you: autodownload the model and build the user interface.

WanGP finetune system can be also used to tweak default models : for instance you can add on top of an existing model some loras that will be always applied transparently.

Finetune models definitions are light json files that can be easily shared. You can find some of them on the WanGP *discord* server https://discord.gg/g7efUW9jGV

All the finetunes definitions files should be stored in the *finetunes/* subfolder.

Finetuned models have been tested so far with Wan2.1 text2video, Wan2.1 image2video,  Hunyuan Video text2video. There isn't currently any support for LTX Video finetunes.



## Create a new Finetune Model Definition
All the finetune models definitions are json files stored in the **finetunes/** sub folder. All the corresponding finetune model weights when they are downloaded will be stored in the *ckpts/* subfolder and will sit next to the base models.

All the models used by WanGP are also described using the finetunes json format and can be found in the **defaults/** subfolder. Please don’t modify any file in the **defaults/** folder.

However you can use these files as starting points for new definition files and to get an idea of the structure of a definition file. If you want to change how a base model is handled (title, default settings, path to model weights, …) you may override any property of the default finetunes definition file by creating a new file in the finetunes folder with the same name. Everything will happen as if the two models will be merged property by property with a higher priority given to the finetunes model definition.

A definition is built from a *settings file* that can contains all the default parameters for a video generation. On top of this file a subtree named **model** contains all the information regarding the finetune (URLs to download model, corresponding base model id, ...).

You can obtain a settings file in several ways:
- In the subfolder **settings**, get the json file that corresponds to the base model of your finetune (see the next section for the list of ids of base models)
- From the user interface, select the base model for which you want to create a finetune and click **export settings**

Here are steps:
1) Create a *settings file*
2) Add a **model** subtree with the finetune description
3) Save this file in the subfolder **finetunes**. The name used for the file will be used as its id. It is a good practise to prefix the name of this file with the base model. For instance for a finetune named **Fast*** based on  Hunyuan Text 2 Video model *hunyuan_t2v_fast.json*. In this example the Id is *hunyuan_t2v_fast*.
4) Restart WanGP

## Architecture Models Ids
A finetune is derived from a base model and will inherit all the user interface and corresponding model capabilities, here are some Architecture Ids:
- *t2v*: Wan 2.1 Video text 2 video
- *i2v*: Wan 2.1 Video image 2 video 480p and 720p
- *vace_14B*: Wan 2.1 Vace 14B
- *hunyuan*: Hunyuan Video text 2 video
- *hunyuan_i2v*: Hunyuan Video image 2 video

Any file name in the defaults subfolder (without the json extension) corresponds to an architecture id.

Please note that weights of some architectures correspond to a combination of weight of a one architecture which are completed by the weights of one more or modules.

A module is a set a weights that are insufficient to be model by itself but that can be added to an existing model to extend its capabilities.

For instance if one adds a module *vace_14B* on top of a model with architecture *t2v* one gets get a model with the *vace_14B* architecture. Here *vace_14B* stands for both an architecture name and a module name. The module system allows you to reuse shared weights between models.


## The Model Subtree
- *name* : name of the finetune used to select
- *architecture* : architecture Id of the base model of the finetune (see previous section)
- *description*: description of the finetune that will appear at the top
- *URLs*: URLs of all the finetune versions (quantized / non quantized). WanGP will pick the version that is the closest to the user preferences. You will need to follow a naming convention to help WanGP identify the content of each version (see next section). WanGP supports 8 bits quantized model that have been quantized using **quanto** and Scaled FP8 models. WanGP offers a command switch to build easily such a quantized model (see below). *URLs* can contain also paths to local file to allow testing.
- *URLs2*: URLs of all the finetune versions (quantized / non quantized) of the weights used for the second phase of a model. For instance with Wan 2.2, the first phase contains the High Noise model weights and the second phase contains the Low Noise model weights. This feature can be used with other models than Wan 2.2 to combine different model weights during the same video generation.
-*text_encoder_URLs* : URLs of the text_encoder versions (quantized or not), if specified will override the default text encoder
- *modules*: this a list of modules to be combined with the models referenced by the URLs. A module is a model extension that is merged with a model to expand its capabilities. Supported models so far are : *vace_14B* and *multitalk*. For instance the full Vace model is the fusion of a Wan text 2 video and the Vace module.
- *preload_URLs* : URLs of files to download no matter what (used to load quantization maps for instance)
-*loras* : URLs of Loras that will applied before any other Lora specified by the user. These loras will be quite often Loras accelerators. For instance if you specify here the FusioniX Lora you will be able to reduce the number of generation steps to 10
be able to reduce the number of generation steps to 10
-*loras_multipliers* : a list of float numbers or strings that defines the weight of each Lora mentioned in *Loras*. The string syntax is used if you want your lora multiplier to change over the steps (please check the Loras doc) or if you want a multiplier to be applied on a specific High Noise phase or Low Noise phase of a Wan 2.2 model. For instance, here the multiplier will be only applied during the High Noise phase and for half of the steps of this phase the multiplier will be 1 and for the other half 1.1.
```
"loras" : [ "my_lora.safetensors"],
"loras_multipliers" : [ "1,1.1;0"]
```

- *auto_quantize*: if set to True and no quantized model URL is provided, WanGP will perform on the fly quantization if the user expects a quantized model
-*visible* : by default assumed to be true. If set to false the model will no longer be visible. This can be useful if you create a finetune to override a default model and hide it.
-*image_outputs* : turn any model that generates a video into a model that generates images. In fact it will adapt the user interface for image generation and ask the model to generate a video with a single frame.
-*video_prompt_enhancer_instructions* : this allows you override the system prompt used by the Prompt Enhancer when generating a Video with this finetune
-*image_prompt_enhancer_instructions* : this allows you override the system prompt used by the Prompt Enhancer when generating an Image with this finetune
-*video_prompt_enhancer_max_tokens*: override for the maximum number of tokens generated by the prompt enhancer when generating a Video (default 256)
-*image_prompt_enhancer_max_tokens*: override for the maximum number of tokens generated by the prompt enhancer when generating an Image (default 256)

In order to favor reusability the properties of *URLs*, *modules*, *loras* and  *preload_URLs* can contain instead of a list of URLs a single text which corresponds to the id of a finetune or default model to reuse. Instead of:
```
    "URLs": [
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_mbf16.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_quanto_mbf16_int8.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_high_quanto_mfp16_int8.safetensors"
    ],
    "URLs2": [
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_mbf16.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_quanto_mbf16_int8.safetensors",
      "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_text2video_14B_low_quanto_mfp16_int8.safetensors"
    ],
```
 You can write:
```
 "URLs":  "t2v_2_2",
 "URLs2":  "t2v_2_2",
```


Example of **model** subtree
```
        "model":
        {
                "name": "Wan text2video FusioniX 14B",
                "architecture" : "t2v",
                "description": "A powerful merged text-to-video model based on the original WAN 2.1 T2V model, enhanced using multiple open-source components and LoRAs to boost motion realism, temporal consistency, and expressive detail. multiple open-source models and LoRAs to boost temporal quality, expressiveness, and motion realism.",
                "URLs": [
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_fp16.safetensors",
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_fp16_int8.safetensors",
                        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan14BT2VFusioniX_quanto_bf16_int8.safetensors"
                ],
        "preload_URLs": [
        ],
                "auto_quantize": true
        },
```

## Finetune Model Naming Convention
If a model is not quantized, it is assumed to be mostly 16 bits (with maybe a few 32 bits weights), so *bf16* or *fp16* should appear somewhere in the name. If you need examples just look at the **ckpts** subfolder, the naming convention for the base models is the same.

If a model is quantized the term *quanto* should also be included since WanGP supports for the moment only *quanto* quantized model, most specically you should replace *fp16* by *quanto_fp16_int8* or *bf6* by *quanto_bf16_int8*.

Please note it is important than *bf16", "fp16* and *quanto* are all in lower cases letters.

## Creating a Quanto Quantized file
If you launch the app with the *--save-quantized* switch, WanGP will create a quantized file in the **ckpts** subfolder just after the model has been loaded. Please note that the model will *bf16* or *fp16* quantized depending on what you chose in the configuration menu.

1) Make sure that in the finetune definition json file there is only a URL or filepath that points to the non quantized model
2) Launch WanGP *python wgp.py --save-quantized*
3) In the configuration menu *Transformer Data Type* property choose either *BF16* of *FP16*
4) Launch a video generation (settings used do not matter). As soon as the model is loaded, a new quantized model will be created in the **ckpts** subfolder if it doesn't already exist.
5) WanGP will update automatically the finetune definition file with the local path of the newly created quantized file (the list "URLs" will have an extra value such as *"ckpts/finetune_quanto_fp16_int8.safetensors"*
6) Remove *--save-quantized*, restart WanGP and select *Scaled Int8 Quantization* in the *Transformer Model Quantization* property
7) Launch a new generation and verify in the terminal window that the right quantized model is loaded
8) In order to share the finetune definition file you will need to store the fine model weights in the cloud. You can upload them for instance on *Huggingface*. You can now replace in the finetune definition file the local path by a URL (on Huggingface to get the URL of the model file click *Copy download link* when accessing the model properties)

You need to create a quantized model specifically for *bf16* or *fp16* as they can not converted on the fly. However there is no need for a non quantized model as they can be converted on the fly while being loaded.

Wan models supports both *fp16* and *bf16* data types albeit *fp16* delivers in theory better quality. On the contrary Hunyuan and LTXV supports only *bf16*.
