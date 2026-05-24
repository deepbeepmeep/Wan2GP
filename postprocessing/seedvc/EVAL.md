### Zero-shot voice conversion🎙🔁
We have performed a series of objective evaluations on our Seed-VC's voice conversion capabilities. 
For ease of reproduction, source audios are 100 random utterances from LibriTTS-test-clean, and reference audios are 12 randomly picked in-the-wild voices with unique characteristics. <br>  

Source audios can be found under `./examples/libritts-test-clean` <br>
Reference audios can be found under `./examples/reference` <br>

We evaluate the conversion results in terms of speaker embedding cosine similarity (SECS), word error rate (WER) and character error rate (CER) and compared
our results with two strong open sourced baselines, namely [OpenVoice](https://github.com/myshell-ai/OpenVoice) and [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).  
Results in the table below shows that our Seed-VC model significantly outperforms the baseline models in both intelligibility and speaker similarity.<br>

| Models\Metrics | SECS↑      | WER↓      | CER↓     | SIG↑     | BAK↑     | OVRL↑    |
|----------------|------------|-----------|----------|----------|----------|----------|
| Ground Truth   | 1.0000     | 8.02      | 1.57     | ~        | ~        | ~        |
| OpenVoice      | 0.7547     | 15.46     | 4.73     | **3.56** | **4.02** | **3.27** |
| CosyVoice      | 0.8440     | 18.98     | 7.29     | 3.51     | **4.02** | 3.21     |
| Seed-VC(Ours)  | **0.8676** | **11.99** | **2.92** | 3.42     | 3.97     | 3.11     |

We have also compared with non-zero-shot voice conversion models for several speakers (based on model availability):

| Characters          | Models\Metrics | SECS↑      | WER↓      | CER↓     | SIG↑     | BAK↑     | OVRL↑    |
|---------------------|----------------|------------|-----------|----------|----------|----------|----------|
| ~                   | Ground Truth   | 1.0000     | 6.43      | 1.00     | ~        | ~        | ~        |
| Tokai Teio          | So-VITS-4.0    | 0.8637     | 21.46     | 9.63     | 3.06     | 3.66     | 2.68     |
|                     | Seed-VC(Ours)  | **0.8899** | **15.32** | **4.66** | **3.12** | **3.71** | **2.72** |
| Milky Green         | So-VITS-4.0    | 0.6850     | 48.43     | 32.50    | 3.34     | 3.51     | 2.82     |
|                     | Seed-VC(Ours)  | **0.8072** | **7.26**  | **1.32** | **3.48** | **4.07** | **3.20** |
| Matikane Tannhuaser | So-VITS-4.0    | 0.8594     | 16.25     | 8.64     | **3.25** | 3.71     | 2.84     |
|                     | Seed-VC(Ours)  | **0.8768** | **12.62** | **5.86** | 3.18     | **3.83** | **2.85** |

Results show that, despite not being trained on the target speakers, Seed-VC is able to achieve significantly better results than the non-zero-shot models. 
However, this may vary a lot depending on the SoVITS model quality. PR or Issue is welcomed if you find this comparison unfair or inaccurate.  
(Tokai Teio model from [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))   
(Matikane Tannhuaser model from [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))  
(Milky Green model from [sparanoid/milky-green-sovits-4](https://huggingface.co/spaces/sparanoid/milky-green-sovits-4))  

*English ASR result computed by [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) model*  
*Speaker embedding computed by [resemblyzer](https://github.com/resemble-ai/Resemblyzer) model* <br>  

You can reproduce the evaluation by running `eval.py` script.  
```bash
python eval.py 
--source ./examples/libritts-test-clean
--target ./examples/reference
--output ./examples/eval/converted
--diffusion-steps 25
--length-adjust 1.0
--inference-cfg-rate 0.7
--xvector-extractor "resemblyzer"
--baseline ""  # fill in openvoice or cosyvoice to compute baseline result
--max-samples 100  # max source utterances to go through
```
Before that, make sure you have openvoice and cosyvoice repo correctly installed on `../OpenVoice/` and `../CosyVoice/` if you would like to run baseline evaluation.

### Zero-shot singing voice conversion🎤🎶

Additional singing voice conversion evaluation is done on [M4Singer](https://github.com/M4Singer/M4Singer) dataset, with 4 target speakers whose audio data is available [here](https://huggingface.co/datasets/XzJosh/audiodataset).  
Speaker similariy is calculated by averaging the cosine similarities between conversion result and all available samples in respective character dataset.   
For each character, one random utterance is chosen as the prompt for zero-shot inference. For comparison, we trained respective [RVCv2-f0-48k](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) model for each character as baseline.  
100 random utterances for each singer type are used as source audio.

| Models\Metrics | F0CORR↑ | F0RMSE↓ | SECS↑      | CER↓      | SIG↑     | BAK↑     | OVRL↑    |
|----------------|---------|---------|------------|-----------|----------|----------|----------|
| RVCv2          | 0.9404  | 30.43   | 0.7264     | 28.46     | **3.41** | **4.05** | **3.12** |
| Seed-VC(Ours)  | 0.9375  | 33.35   | **0.7405** | **19.70** | 3.39     | 3.96     | 3.06     |

<details>
<summary>Click to expand detailed evaluation results</summary>

| Source Singer Type | Characters         | Models\Metrics | F0CORR↑ | F0RMSE↓ | SECS↑      | CER↓      | SIG↑ | BAK↑ | OVRL↑    |
|--------------------|--------------------|----------------|---------|---------|------------|-----------|------|------|----------|
| Alto (Female)      | ~                  | Ground Truth   | 1.0000  | 0.00    | ~          | 8.16      | ~    | ~    | ~        |
|                    | Azuma (Female)     | RVCv2          | 0.9617  | 33.03   | **0.7352** | 24.70     | 3.36 | 4.07 | 3.07     |
|                    |                    | Seed-VC(Ours)  | 0.9658  | 31.64   | 0.7341     | **15.23** | 3.37 | 4.02 | 3.07     |
|                    | Diana (Female)     | RVCv2          | 0.9626  | 32.56   | 0.7212     | 19.67     | 3.45 | 4.08 | **3.17** |
|                    |                    | Seed-VC(Ours)  | 0.9648  | 31.94   | **0.7457** | **16.81** | 3.49 | 3.99 | 3.15     |
|                    | Ding Zhen (Male)   | RVCv2          | 0.9013  | 26.72   | 0.7221     | 18.53     | 3.37 | 4.03 | 3.06     |
|                    |                    | Seed-VC(Ours)  | 0.9356  | 21.87   | **0.7513** | **15.63** | 3.44 | 3.94 | **3.09** |
|                    | Kobe Bryant (Male) | RVCv2          | 0.9215  | 23.90   | 0.7495     | 37.23     | 3.49 | 4.06 | **3.21** |
|                    |                    | Seed-VC(Ours)  | 0.9248  | 23.40   | **0.7602** | **26.98** | 3.43 | 4.02 | 3.13     |
| Bass (Male)        | ~                  | Ground Truth   | 1.0000  | 0.00    | ~          | 8.62      | ~    | ~    | ~        |
|                    | Azuma              | RVCv2          | 0.9288  | 32.62   | **0.7148** | 24.88     | 3.45 | 4.10 | **3.18** |
|                    |                    | Seed-VC(Ours)  | 0.9383  | 31.57   | 0.6960     | **10.31** | 3.45 | 4.03 | 3.15     |
|                    | Diana              | RVCv2          | 0.9403  | 30.00   | 0.7010     | 14.54     | 3.53 | 4.15 | **3.27** |
|                    |                    | Seed-VC(Ours)  | 0.9428  | 30.06   | **0.7299** | **9.66**  | 3.53 | 4.11 | 3.25     |
|                    | Ding Zhen          | RVCv2          | 0.9061  | 19.53   | 0.6922     | 25.99     | 3.36 | 4.09 | **3.08** |
|                    |                    | Seed-VC(Ours)  | 0.9169  | 18.15   | **0.7260** | **14.13** | 3.38 | 3.98 | 3.07     |
|                    | Kobe Bryant        | RVCv2          | 0.9302  | 16.37   | 0.7717     | 41.04     | 3.51 | 4.13 | **3.25** |
|                    |                    | Seed-VC(Ours)  | 0.9176  | 17.93   | **0.7798** | **24.23** | 3.42 | 4.08 | 3.17     |
| Soprano (Female)   | ~                  | Ground Truth   | 1.0000  | 0.00    | ~          | 27.92     | ~    | ~    | ~        |
|                    | Azuma              | RVCv2          | 0.9742  | 47.80   | 0.7104     | 38.70     | 3.14 | 3.85 | **2.83** |
|                    |                    | Seed-VC(Ours)  | 0.9521  | 64.00   | **0.7177** | **33.10** | 3.15 | 3.86 | 2.81     |
|                    | Diana              | RVCv2          | 0.9754  | 46.59   | **0.7319** | 32.36     | 3.14 | 3.85 | **2.83** |
|                    |                    | Seed-VC(Ours)  | 0.9573  | 59.70   | 0.7317     | **30.57** | 3.11 | 3.78 | 2.74     |
|                    | Ding Zhen          | RVCv2          | 0.9543  | 31.45   | 0.6792     | 40.80     | 3.41 | 4.08 | **3.14** |
|                    |                    | Seed-VC(Ours)  | 0.9486  | 33.37   | **0.6979** | **34.45** | 3.41 | 3.97 | 3.10     |
|                    | Kobe Bryant        | RVCv2          | 0.9691  | 25.50   | 0.6276     | 61.59     | 3.43 | 4.04 | **3.15** |
|                    |                    | Seed-VC(Ours)  | 0.9496  | 32.76   | **0.6683** | **39.82** | 3.32 | 3.98 | 3.04     |
| Tenor (Male)       | ~                  | Ground Truth   | 1.0000  | 0.00    | ~          | 5.94      | ~    | ~    | ~        |
|                    | Azuma              | RVCv2          | 0.9333  | 42.09   | **0.7832** | 16.66     | 3.46 | 4.07 | **3.18** |
|                    |                    | Seed-VC(Ours)  | 0.9162  | 48.06   | 0.7697     | **8.48**  | 3.38 | 3.89 | 3.01     |
|                    | Diana              | RVCv2          | 0.9467  | 36.65   | 0.7729     | 15.28     | 3.53 | 4.08 | **3.24** |
|                    |                    | Seed-VC(Ours)  | 0.9360  | 41.49   | **0.7920** | **8.55**  | 3.49 | 3.93 | 3.13     |
|                    | Ding Zhen          | RVCv2          | 0.9197  | 22.82   | 0.7591     | 12.92     | 3.40 | 4.02 | **3.09** |
|                    |                    | Seed-VC(Ours)  | 0.9247  | 22.77   | **0.7721** | **13.95** | 3.45 | 3.82 | 3.05     |
|                    | Kobe Bryant        | RVCv2          | 0.9415  | 19.33   | 0.7507     | 30.52     | 3.48 | 4.02 | **3.19** |
|                    |                    | Seed-VC(Ours)  | 0.9082  | 24.86   | **0.7764** | **13.35** | 3.39 | 3.93 | 3.07     |
</details>
  
  
Despite Seed-VC is not trained on the target speakers, and only one random utterance is used as prompt, it still constantly outperforms speaker-specific RVCv2 models 
in terms of speaker similarity (SECS) and intelligibility (CER), which demonstrates the superior voice cloning capability and robustness of Seed-VC.   

However, it is observed that Seed-VC's audio quality (DNSMOS) is slightly lower than RVCv2. We take this drawback seriously and 
will give high priority to improve the audio quality in the future.  
PR or issue is welcomed if you find this comparison unfair or inaccurate.

*Chinese ASR result computed by [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)*  
*Speaker embedding computed by [resemblyzer](https://github.com/resemble-ai/Resemblyzer) model*  
*We set +12 semitones pitch shift for male-to-female conversion and -12 semitones for female-to-male converison, otherwise 0 pitch shift*

