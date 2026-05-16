# Seed-VC  
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  [![arXiv](https://img.shields.io/badge/arXiv-2411.09943-<COLOR>.svg)](https://arxiv.org/abs/2411.09943)

*English | [简体中文](README-ZH.md) | [日本語](README-JA.md)*  

[real-time-demo.webm](https://github.com/user-attachments/assets/86325c5e-f7f6-4a04-8695-97275a5d046c)

目前发布的模型支持 *零样本语音转换* 🔊 、*零样本实时语音转换* 🗣️ 和 *零样本歌声转换* 🎶。无需任何训练，只需1~30秒的参考语音，即可克隆声音。

我们支持进一步使用自定义数据进行微调，以提高特定说话人的性能，数据需求门槛极低 **（每位说话人至少1条语音）** ，训练速度极快 **（最少100步，在T4上只需2分钟）**！

**实时语音转换** 支持约300ms的算法延迟和约100ms的设备侧延迟，适用于在线会议、游戏和直播。

要查看演示和与之前语音转换模型的比较，请访问我们的[演示页面](https://plachtaa.github.io/seed-vc/)🌐 和 [评估结果](EVAL.md)📊。

我们会不断改进模型质量并增加更多功能。

## 评估📊
查看 [EVAL.md](EVAL.md) 获取客观评估结果和与其他基准模型的比较。

## 使用🛠️
我们已发布用于不同目的的3个模型：

| 版本   | 模型名称                                                                                                                                                                                                                       | 用途         | 采样率   | Content编码器    | 声码器     | 隐藏层维度 | 层数 | 参数量  | 备注                 |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-------|---------------|---------|-------|----|------|--------------------|
| v1.0 | seed-uvit-tat-xlsr-tiny ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_uvit_tat_xlsr_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml))                                                     | 声音转换 (VC)  | 22050 | XLSR-large    | HIFT    | 384   | 9  | 25M  | 适合实时语音转换           |
| v1.0 | seed-uvit-whisper-small-wavenet ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml)) | 声音转换 (VC)  | 22050 | Whisper-small | BigVGAN | 512   | 13 | 98M  | 性能更好但推理稍慢，适合离线语音转换 |
| v1.0 | seed-uvit-whisper-base ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml))       | 歌声转换 (SVC) | 44100 | Whisper-small | BigVGAN | 768   | 17 | 200M | 强大的零样本推理能力，用于歌声转换  |

首次推理时将自动下载最新模型的检查点。 如果因网络原因无法访问 Hugging Face，请尝试在每个命令前添加 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站。

命令行推理：
```bash
python inference.py --source <source-wav>
--target <referene-wav>
--output <output-dir>
--diffusion-steps 25 # 推荐为歌声转换设置为30~50
--length-adjust 1.0
--inference-cfg-rate 0.7
--f0-condition False # 设置为 True 进行歌声转换
--auto-f0-adjust False # 设置为 True 自动调整源音高至目标音高，通常不用于歌声转换（会导致歌声与BGM调性不一致）
--semi-tone-shift 0 # 歌声转换中的音高移位（半音）
--checkpoint <path-to-checkpoint>
--config <path-to-config>
```
参数说明：
- `source` 要转换为参考声音的语音文件路径
- `target` 作为声音参考的语音文件路径
- `output` 输出目录的路径
- `diffusion-steps` 使用的扩散步数，默认为 25，质量最佳使用 30-50，最快推理使用 4-10
- `length-adjust` 长度调整因子，默认值为 1.0，设置 <1.0 加速语音，>1.0 减慢语音
- `inference-cfg-rate` classifier free guidance rate，默认为 0.7
- `f0-condition` 是否对输出音高进行调节，默认为 False，设置为 True 用于歌声转换
- `auto-f0-adjust` 是否自动调整源音高到目标音高，默认为 False，通常不用于歌声转换
- `semi-tone-shift` 歌声转换中的音高移位（半音），默认值为 0
- `checkpoint` 如果已训练或微调自己的模型，请指定模型检查点路径，若留空将自动下载 Hugging Face 的默认模型(`seed-uvit-whisper-small-wavenet` if `f0-condition` is `False` else `seed-uvit-whisper-base`)
- `config` 如果已训练或微调自己的模型，请指定模型配置文件路径，若留空将自动下载 Hugging Face 的默认配置 


语音转换 Web UI:
```bash
python app_vc.py --checkpoint <path-to-checkpoint> --config <path-to-config>
```
- `checkpoint` 模型检查点路径，若为空将自动下载默认模型 (`seed-uvit-whisper-small-wavenet`)
- `config` 模型配置文件路径，若为空将自动下载默认配置

然后在浏览器中打开 `http://localhost:7860/` 使用 Web 界面。

运行命令前先设置环境变量:
`export export HUGGING_FACE_HUB_TOKEN={从https://huggingface.co/settings/tokens获取}`

歌声转换 Web UI:
```bash
python app_svc.py --checkpoint <path-to-checkpoint> --config <path-to-config>
```
- `checkpoint` 模型检查点路径，若为空将自动下载默认模型 (`seed-uvit-whisper-base`)
- `config` 模型配置文件路径，若为空将自动下载默认配置  

集成 Web UI:
```bash
python app.py
```
此命令将仅加载预训练模型进行零样本推理。要使用自定义检查点，请按上述步骤运行 `app_vc.py` 或 `app_svc.py`。

实时语音转换 GUI:
```bash
python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
```
- `checkpoint` 模型检查点路径，若为空将自动下载默认模型 (`seed-uvit-tat-xlsr-tiny`)
- `config` 模型配置文件路径，若为空将自动下载默认配置  

重要提示: 强烈建议使用 GPU 进行实时语音转换。 在 NVIDIA RTX 3060 笔记本 GPU 上进行了一些性能测试，结果和推荐参数设置如下：

| 模型配置                | 扩散步数 | Inference CFG Rate | 最大prompt长度 | 每块时间 (s) | 交叉淡化长度 (s) | 额外上下文（左）(s) | 额外上下文（右）(s) | 延迟 (ms） | 每块推理时间 (ms) |
|---------------------|------|--------------------|------------|----------|------------|-------------|-------------|---------|-------------| 
| seed-uvit-xlsr-tiny | 10   | 0.7                | 3.0        | 0.18s    | 0.04s      | 2.5s        | 0.02s       | 430ms   | 150ms       |

你可以根据设备性能调整 GUI 中的参数，只要推理时间小于块时间，语音转换流就可以正常工作。 注意，如果你正在运行其他占用 GPU 的任务（如游戏、看视频），推理速度可能会下降。

实时转换界面的参数说明：
- `Diffusion Steps` 是扩散步数，推荐实时转换设置为4~10；
- `Inference CFG Rate` 是classifier free guidance rate，默认0.7，设置为0.0可以获得1.5x的加速；
- `Max Prompt Length` 是最大音频提示长度，设置为较低值可以加快推理速度，但可能会降低与提示语音的相似度；
- `Block Time` 是每块时间，值越高延迟越高，该值必须大于每块推理时间，根据硬件条件设置；
- `Crossfade Length` 是交叉淡化长度，通常不需要更改；
- `Extra context (left)` 是推理的额外上下文，设置为较高值可以增加稳定性，但会增加每块推理时间；
- `Extra context (right)` 是推理的额外上下文，设置为较高值可以增加稳定性，但会增加每块推理时间以及延迟；

算法延迟大约为 `Block Time * 2 + Extra context (right)`，设备侧延迟通常为100ms左右。总体延迟为两者之和。

你可以使用 [VB-CABLE](https://vb-audio.com/Cable/) 将变声器输出映射到一个虚拟麦克风上，以便其它应用读取.  

*(GUI and audio chunking logic are modified from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), thanks for their brilliant implementation!)*

## 训练🏋️
在自定义数据上进行微调可以让模型更精确地克隆某个人的声音。这将大幅提高特定说话人的相似度，但可能会略微增加 WER（词错误率）。  
这里是一个简单的Colab示例以供参考: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R1BJTqMsTXZzYAVx3j1BiemFXog9pbQG?usp=sharing)
1. 准备您的数据集。必须满足以下要求：
    - 文件结构不重要
    - 每条音频长度必须在1-30秒之间，否则会被自动忽略
    - 所有音频文件必须是以下格式之一：`.wav` `.flac` `.mp3` `.m4a` `.opus` `.ogg`
    - 不需要说话人标签，但请确保每位说话人至少有 1 条语音
    - 当然，数据越多，模型的表现就越好
    - 训练样本应该选择尽量干净，不带背景音乐或噪音的音频
2. 从 `configs/presets/` 中选择一个模型配置文件进行微调，或者创建自己的配置文件从头开始训练。
    - 对于微调，可以选择以下配置之一：
        - `./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml` 用于实时语音转换
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml` 用于离线语音转换
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` 用于歌声转换
3. 运行以下命令开始训练：
```bash
python train.py 
--config <path-to-config> 
--dataset-dir <path-to-data>
--run-name <run-name>
--batch-size 2
--max-steps 1000
--max-epochs 1000
--save-every 500
--num-workers 0
```
where:
- `config` 模型配置文件路径，选择上面之一进行微调，或者创建自己的配置文件从头开始训练
- `dataset-dir` 数据集目录路径，应为包含所有音频文件的文件夹
- `run-name` 运行名称，用于保存模型检查点和日志
- `batch-size` 训练的批大小，根据 GPU 内存选择
- `max-steps` 最大训练步数，取决于数据集大小和训练时间
- `max-epochs` 最大训练轮数，取决于数据集大小和训练时间
- `save-every` 保存模型检查点的步数
- `num-workers` 数据加载的工作线程数量，建议 Windows 上设置为 0

4. 如果需要从上次停止的地方继续训练，只需运行同样的命令即可。通过传入相同的 `run-name` 和 `config` 参数，程序将能够找到上次训练的检查点和日志。

5. 训练完成后，您可以通过指定检查点和配置文件的路径来进行推理。
    - 它们应位于 `./runs/<run-name>/` 下，检查点命名为 `ft_model.pth`，配置文件名称与训练配置文件相同。
    - 在推理时，您仍需指定要使用的说话人的参考音频文件，类似于零样本推理。

## TODO📝
- [x] 发布代码
- [x] 发布预训练模型： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Hugging Face Space 演示： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTML 演示页面： [Demo](https://plachtaa.github.io/seed-vc/)
- [x] 流式推理
- [x] 降低延迟
- [x] 实时变声Demo视频
- [x] 歌声转换
- [x] 提高源音频抗噪性
- [ ] 潜在的架构改进
    - [x] 类似U-ViT 的skip connection
    - [x] 将输入更改为 OpenAI Whisper
    - [x] Time as Token
- [x] 自定义数据训练代码
- [x] 单样本/少样本说话人微调
- [x] 歌声解码器更改为 NVIDIA 的 BigVGAN
- [x] 44k Hz 歌声转换模型
- [x] 歌声转换的客观指标评估以及与RVC/SoVITS模型的比较
- [x] 提升音质
- [ ] 用于改善歌声转换的NSF歌声解码器
- [x] 实时变声脚本添加了VAD模型，避免没有说话时模型输出杂音
- [x] Google Colab 笔记本训练脚本以及样例
- [ ] 替换whisper为更先进的语义内容提取器
- [ ] 更多待添加

## 更新日志 🗒️
- 2024-11-26:
    - 更新 v1.0 更小版本的预训练模型，优化实时语音转换
    - 支持单样本/少样本的单/多说话人微调
    - 支持在 WebUI 和实时变声 GUI 中使用自定义检查点
- 2024-11-19:
    - paper已提交至arXiv
- 2024-10-27:
    - 更新了实时变声脚本
- 2024-10-25:
    - 添加了详尽的歌声转换评估结果以及与RVCv2模型的比较
- 2024-10-24:
    - 更新了44kHz歌声转换模型
- 2024-10-07:
    - 更新了 v0.3 预训练模型，将语音内容编码器更改为 OpenAI Whisper
    - 添加了 v0.3 预训练模型的客观指标评估结果
- 2024-09-22:
    - 将歌声转换模型的解码器更改为 BigVGAN，解决了大部分高音部分无法正确转换的问题
    - 在Web UI中支持对长输入音频的分段处理以及流式输出
- 2024-09-18:
    - 更新了用于歌声转换的模型
- 2024-09-14:
    - 更新了 v0.2 预训练模型，具有更小的尺寸和更少的扩散步骤即可达到相同质量，且增加了控制韵律保留的能力
    - 添加了命令行推理脚本
    - 添加了安装和使用说明

## 鸣谢🙏
- [Amphion](https://github.com/open-mmlab/Amphion) for providing computational resources and inspiration!
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for foundationing the real-time voice conversion
- [SEED-TTS](https://arxiv.org/abs/2406.02430) for the initial idea