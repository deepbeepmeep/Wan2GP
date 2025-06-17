# FINETUNES

A Finetuned model is model that shares the same architecture of one specific model but has derived weights from this model. Some finetuned models have been created by combining multiple finetuned models.

As there are potentially an infinite number of finetunes, specific finetuned models are not known by default by WanGP, however you can create a finetuned model definition that will tell WanGP about the existence of this finetuned model and WanGP will do as usual all the work for you: autodownload the model and build the user interface.

Finetune models definitions are light json files that can be easily shared. You can find some of them on the WanGP *discord* server https://discord.gg/g7efUW9jGV

Finetuned models have been tested so far with Wan2.1 text2video, Wan2.1 image2video,  Hunyuan Video text2video. There isn't currently any support for LTX Video finetunes.

## Create a new Finetune Model Definition
All the finetune models definitions are json files stored in the **finetunes** sub folder. All the corresponding finetune model weights will be stored in the *ckpts* subfolder and will sit next to the base models.

WanGP comes with a few prebuilt finetune models that you can use as starting points and to get an idea of the structure of the definition file.

A definition is built from a *settings file* that can contains all the default parameters for a video generation. On top of this file a subtree named **model** contains all the information regarding the finetune (URLs to download model, corresponding base model id, ...).

You can obtain a settings file in several ways:
- In the subfolder **settings**, get the json file that corresponds to the base model of your finetune (see the next section for the list of ids of base models)
- From the user interface, go to the base model and click **export settings**

Here are steps:
1) Create a *settings file*
2) Add a **model** subtree with the finetune description
3) Save this file in the subfolder **finetunes**. The name used for the file will be used as its id. It is a good practise to prefix the name of this file with the base model. For instance for a finetune named **Fast*** based on  Hunyuan Text 2 Video model *hunyuan_t2v_fast.json*. In this example the Id is *hunyuan_t2v_fast*.
4) Restart WanGP

## Architecture Models Ids
A finetune is derived from a base model and will inherit all the user interface and corresponding model capabilities, here are Architecture Ids:
- *t2v*: Wan 2.1 Video text 2 
- *i2v*: Wan 2.1 Video image 2 480p
- *i2v_720p*: Wan 2.1 Video image 2 720p
- *vace_14B*: Wan 2.1 Vace 14B
- *hunyuan*: Hunyuan Video text 2 video
- *hunyuan_i2v*: Hunyuan Video image 2 video

## The Model Subtree
- *name* : name of the finetune used to select
- *architecture* : architecture Id of the base model of the finetune (see previous section)
- *description*: description of the finetune that will appear at the top
- *URLs*: URLs of all the finetune versions (quantized / non quantized). WanGP will pick the version that is the closest to the user preferences. You will need to follow a naming convention to help WanGP identify the content of each version (see next section). Right now WanGP supports only 8 bits quantized model that have been quantized using **quanto**. WanGP offers a command switch to build easily such a quantized model (see below). *URLs* can contain also paths to local file to allow testing.
- *modules*: this a list of modules to be combined with the models referenced by the URLs. A module is a model extension that is merged with a model to expand its capabilities. So far the only module supported is Vace 14B  (its id is *vace_14B*). For instance the full Vace model is the fusion of a Wan text 2 video and the Vace module.
- *preload_URLs* : URLs of files to download no matter what (used to load quantization maps for instance)
- *auto_quantize*: if set to True and no quantized model URL is provided, WanGP will perform on the fly quantization if the user expects a quantized model

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