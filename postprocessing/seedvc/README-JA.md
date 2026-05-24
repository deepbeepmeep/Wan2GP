# Seed-VC
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  [![arXiv](https://img.shields.io/badge/arXiv-2411.09943-<COLOR>.svg)](https://arxiv.org/abs/2411.09943)

*[English](README.md) | [简体中文](README-ZH.md) | 日本語*

[real-time-demo.webm](https://github.com/user-attachments/assets/86325c5e-f7f6-4a04-8695-97275a5d046c)

*(注意：この文書は機械翻訳によって生成されたものです。正確性を確保するよう努めていますが、不明確な点がございましたら英語版をご参照ください。翻訳の改善案がございましたら、PRを歓迎いたします。)*

現在リリースされているモデルは、*ゼロショット音声変換* 🔊、*ゼロショットリアルタイム音声変換* 🗣️、*ゼロショット歌声変換* 🎶 に対応しています。トレーニングなしで、1〜30秒の参照音声からボイスクローニングが可能です。

カスタムデータでの追加ファインチューニングをサポートしており、特定の話者/話者群に対するパフォーマンスを向上させることができます。データ要件は極めて少なく（**話者あたり最低1発話**）、トレーニング速度も非常に速い（**最低100ステップ、T4で2分**）です！

**リアルタイム音声変換**に対応しており、アルゴリズムの遅延は約300ms、デバイス側の遅延は約100msで、オンライン会議、ゲーム、ライブ配信に適しています。

デモや以前の音声変換モデルとの比較については、[デモページ](https://plachtaa.github.io/seed-vc/)🌐と[評価](EVAL.md)📊をご覧ください。

モデルの品質向上と機能追加を継続的に行っています。

## 評価📊
客観的評価結果と他のベースラインとの比較については[EVAL.md](EVAL.md)をご覧ください。

## インストール📥
Windows または Linux で Python 3.10 を推奨します。
```bash
pip install -r requirements.txt
```

## 使用方法🛠️
目的に応じて3つのモデルをリリースしています：

| バージョン | 名称                                                                                                                                                                                                                       | 目的                        | サンプリングレート | コンテンツエンコーダ | ボコーダ | 隠れ次元 | レイヤー数 | パラメータ数 | 備考                                                |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|---------------|-----------------|---------|------------|----------|--------|--------------------------------------------------------|
| v1.0    | seed-uvit-tat-xlsr-tiny ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_uvit_tat_xlsr_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml))                                                     | 音声変換 (VC)          | 22050         | XLSR-large      | HIFT    | 384        | 9        | 25M    | リアルタイム音声変換に適しています                |
| v1.0    | seed-uvit-whisper-small-wavenet ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml)) | 音声変換 (VC)          | 22050         | Whisper-small   | BigVGAN | 512        | 13       | 98M    | オフライン音声変換に適しています                  |
| v1.0    | seed-uvit-whisper-base ([🤗](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth)[📄](configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml))       | 歌声変換 (SVC) | 44100         | Whisper-small   | BigVGAN | 768        | 17       | 200M   | 強力なゼロショットパフォーマンス、歌声変換 |

最新のモデルリリースのチェックポイントは、最初の推論実行時に自動的にダウンロードされます。
ネットワークの理由でhuggingfaceにアクセスできない場合は、すべてのコマンドの前に `HF_ENDPOINT=https://hf-mirror.com` を追加してミラーを使用してください。

コマンドライン推論：
```bash
python inference.py --source <source-wav>
--target <referene-wav>
--output <output-dir>
--diffusion-steps 25 # 歌声変換には30〜50を推奨
--length-adjust 1.0
--inference-cfg-rate 0.7
--f0-condition False # 歌声変換の場合はTrueに設定
--auto-f0-adjust False # ソースピッチをターゲットピッチレベルに自動調整する場合はTrue、通常は歌声変換では使用しない
--semi-tone-shift 0 # 歌声変換のピッチシフト（半音単位）
--checkpoint <path-to-checkpoint>
--config <path-to-config>
--fp16 True
```
各パラメータの説明：
- `source` は変換したい音声ファイルのパス
- `target` は参照音声ファイルのパス
- `output` は出力ディレクトリのパス
- `diffusion-steps` は拡散ステップ数、デフォルトは25、最高品質には30-50、最速推論には4-10を使用
- `length-adjust` は長さ調整係数、デフォルトは1.0、<1.0で音声短縮、>1.0で音声伸長
- `inference-cfg-rate` は出力に微妙な違いをもたらす、デフォルトは0.7
- `f0-condition` はソース音声のピッチを出力に条件付けするフラグ、デフォルトはFalse、歌声変換の場合はTrue
- `auto-f0-adjust` はソースピッチをターゲットピッチレベルに自動調整するフラグ、デフォルトはFalse、通常は歌声変換では使用しない
- `semi-tone-shift` は歌声変換のピッチシフト（半音単位）、デフォルトは0
- `checkpoint` は独自のモデルをトレーニングまたはファインチューニングした場合のモデルチェックポイントへのパス、空白の場合はhuggingfaceからデフォルトモデルを自動ダウンロード（`f0-condition`が`False`の場合は`seed-uvit-whisper-small-wavenet`、それ以外は`seed-uvit-whisper-base`）
- `config` は独自のモデルをトレーニングまたはファインチューニングした場合のモデル設定へのパス、空白の場合はhuggingfaceからデフォルト設定を自動ダウンロード
- `fp16` はfloat16推論を使用するフラグ、デフォルトはTrue

音声変換Web UI：
```bash
python app_vc.py --checkpoint <path-to-checkpoint> --config <path-to-config> --fp16 True
```
- `checkpoint` は独自のモデルをトレーニングまたはファインチューニングした場合のモデルチェックポイントへのパス、空白の場合はhuggingfaceからデフォルトモデルを自動ダウンロード（`seed-uvit-whisper-small-wavenet`）
- `config` は独自のモデルをトレーニングまたはファインチューニングした場合のモデル設定へのパス、空白の場合はhuggingfaceからデフォルト設定を自動ダウンロード

ブラウザで`http://localhost:7860/`にアクセスしてWebインターフェースを使用できます。

歌声変換Web UI：
```bash
python app_svc.py --checkpoint <path-to-checkpoint> --config <path-to-config> --fp16 True
```
- `checkpoint` は独自のモデルをトレーニングまたはファインチューニングした場合のモデルチェックポイントへのパス、空白の場合はhuggingfaceからデフォルトモデルを自動ダウンロード（`seed-uvit-whisper-base`）
- `config` は独自のモデルをトレーニングまたはファインチューニングした場合のモデル設定へのパス、空白の場合はhuggingfaceからデフォルト設定を自動ダウンロード

統合Web UI：
```bash
python app.py
```
これはゼロショット推論用の事前学習済みモデルのみを読み込みます。カスタムチェックポイントを使用する場合は、上記の`app_vc.py`または`app_svc.py`を実行してください。

リアルタイム音声変換GUI：
```bash
python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
```
- `checkpoint` は独自のモデルをトレーニングまたはファインチューニングした場合のモデルチェックポイントへのパス、空白の場合はhuggingfaceからデフォルトモデルを自動ダウンロード（`seed-uvit-tat-xlsr-tiny`）
- `config` は独自のモデルをトレーニングまたはファインチューニングした場合のモデル設定へのパス、空白の場合はhuggingfaceからデフォルト設定を自動ダウンロード

重要：リアルタイム音声変換にはGPUの使用を強く推奨します。
NVIDIA RTX 3060ノートパソコンGPUでいくつかのパフォーマンステストを行い、結果と推奨パラメータ設定を以下に示します：

| モデル構成             | 拡散ステップ | 推論CFGレート | 最大プロンプト長 | ブロック時間 (秒) | クロスフェード長 (秒) | 追加コンテキスト (左) (秒) | 追加コンテキスト (右) (秒) | レイテンシ (ミリ秒) | チャンクあたりの推論時間 (ミリ秒) |
|---------------------------------|-----------------|--------------------|-------------------|----------------|----------------------|--------------------------|---------------------------|--------------|-------------------------------| 
| seed-uvit-xlsr-tiny             | 10              | 0.7                | 3.0               | 0.18           | 0.04                | 2.5                      | 0.02                     | 430          | 150                          |

GUIでパラメータを自身のデバイスのパフォーマンスに合わせて調整できます。推論時間がブロック時間より短ければ、音声変換ストリームは正常に動作するはずです。
他のGPU集約型タスク（ゲーム、動画視聴など）を実行している場合、推論速度が低下する可能性があることに注意してください。

リアルタイム音声変換GUIのパラメータ説明：
- `Diffusion Steps` は拡散ステップ数、リアルタイム変換の場合は通常4~10で最速推論
- `Inference CFG Rate` は出力に微妙な違いをもたらす、デフォルトは0.7、0.0に設定すると1.5倍の推論速度が向上
- `Max Prompt Length` は最大プロンプト長、設定を低くすると推論速度が速くなるが、提示音声との類似性が低下する可能性がある
- `Block Time` は推論の各オーディオ チャンクの時間長です。値が大きいほどレイテンシが長くなります。この値はブロックあたりの推論時間よりも長くする必要があることに注意してください。ハードウェアの状態に応じて設定します。
- `Crossfade Length` はクロスフェード長、通常は変更しない
- `Extra context (left)` は推論のための追加履歴コンテキストの時間長です。値が高いほど推論時間は長くなりますが、安定性は向上します。
- `Extra context (right)` は推論のための追加未来コンテキストの時間長です。値が高いほど推論時間とレイテンシは長くなりますが、安定性は向上します。

アルゴリズムレイテンシーは`Block Time * 2 + Extra context (right)`で、デバイス側レイテンシーは通常100ms程度です。全体の遅延は 2 つの合計です。

[VB-CABLE](https://vb-audio.com/Cable/)を使用して、GUI出力ストリームを仮想マイクにルーティングすることができます。

*（GUIとオーディオチャンキングのロジックは[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)から修正されています。素晴らしい実装に感謝します！）*

## トレーニング🏋️
カスタムデータでのファインチューニングにより、より正確に声をクローニングすることができます。特定の話者に対する話者類似性が大幅に向上しますが、WERが若干上昇する可能性があります。
以下のColabチュートリアルで手順を確認できます：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R1BJTqMsTXZzYAVx3j1BiemFXog9pbQG?usp=sharing)

1. 独自のデータセットを準備します。以下の条件を満たす必要があります：
    - ファイル構造は問いません
    - 各音声ファイルは1〜30秒の範囲である必要があり、それ以外は無視されます
    - すべての音声ファイルは以下のいずれかの形式である必要があります：`.wav` `.flac` `.mp3` `.m4a` `.opus` `.ogg`
    - 話者ラベルは必須ではありませんが、各話者に少なくとも1つの発話があることを確認してください
    - もちろん、データが多いほどモデルのパフォーマンスは向上します
    - トレーニングデータはできるだけクリーンである必要があり、BGMやノイズは望ましくありません

2. ファインチューニング用に`configs/presets/`からモデル設定ファイルを選択するか、ゼロからトレーニングするための独自の設定を作成します。
    - ファインチューニングの場合は、以下のいずれかを選択します：
        - `./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml` リアルタイム音声変換用
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml` オフライン音声変換用
        - `./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` 歌声変換用

3. 以下のコマンドでトレーニングを開始します：
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
各パラメータの説明：
- `config` はモデル設定へのパス、ファインチューニング用に上記のいずれかを選択するか、ゼロからトレーニングする場合は独自の設定を作成
- `dataset-dir` はデータセットディレクトリへのパス、すべての音声ファイルを含むフォルダである必要があります
- `run-name` は実行名で、モデルチェックポイントとログの保存に使用されます
- `batch-size` はトレーニング用のバッチサイズで、GPUメモリに応じて選択します
- `max-steps` は最大トレーニングステップ数で、データセットサイズとトレーニング時間に応じて選択します
- `max-epochs` は最大エポック数で、データセットサイズとトレーニング時間に応じて選択します
- `save-every` はモデルチェックポイントを保存するステップ間隔
- `num-workers` はデータ読み込みのワーカー数、Windowsの場合は0に設定

4. トレーニングが予期せず停止した場合、同じコマンドを再度実行することで、最後のチェックポイントから再開できます（最新のチェックポイントを見つけられるように、`run-name`と`config`引数が同じであることを確認してください）。

5. トレーニング後、チェックポイントと設定ファイルのパスを指定することで、トレーニングしたモデルを推論に使用できます。
    - これらは`./runs/<run-name>/`の下にあり、チェックポイントは`ft_model.pth`という名前で、設定ファイルはトレーニング設定ファイルと同じ名前です。
    - 推論時には、ゼロショット使用時と同様に、使用したい話者の参照音声ファイルを指定する必要があります。

## TODO📝
- [x] コードのリリース
- [x] 事前学習済みモデルのリリース：[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Huggingfaceスペースデモ：[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTMLデモページ：[Demo](https://plachtaa.github.io/seed-vc/)
- [x] ストリーミング推論
- [x] ストリーミング推論のレイテンシー削減
- [x] リアルタイム音声変換のデモ動画
- [x] 歌声変換
- [x] ソース音声のノイズ耐性
- [ ] アーキテクチャの潜在的な改善
    - [x] U-ViTスタイルのスキップ接続
    - [x] OpenAI Whisperへの入力変更
    - [x] Time as Token
- [x] カスタムデータでのトレーニングコード
- [x] フューショット/ワンショット話者ファインチューニング
- [x] 歌声デコーディング用にNVIDIAのBigVGANに変更
- [x] 歌声変換用のWhisperバージョンモデル
- [x] 歌声変換のRVC/SoVITSとの客観的評価と比較
- [x] 音声品質の向上
- [ ] より良い歌声変換のためのNSFボコーダ
- [x] 非発話時のリアルタイム音声変換アーティファクトの修正（VADモデルの追加により対応）
- [x] ファインチューニング例のColabノートブック
- [ ] Whisperをより高度な意味抽出器に置き換える
- [ ] 今後追加予定

## 更新履歴🗒️
- 2024-11-26:
    - リアルタイム音声変換用に最適化されたv1.0 tinyバージョンの事前学習済みモデルを更新
    - ワンショット/フューショットの単一/複数話者ファインチューニングをサポート
    - webUIおよびリアルタイムGUIでカスタムチェックポイントの使用をサポート
- 2024-11-19:
    - arXiv論文公開
- 2024-10-28:
    - より良い音声品質のファインチューニングされた44k歌声変換モデルを更新
- 2024-10-27:
    - リアルタイム音声変換GUIを追加
- 2024-10-25:
    - 歌声変換のRVCv2との包括的な評価結果と比較を追加
- 2024-10-24:
    - 音声コンテンツ入力としてOpenAI Whisperを使用した44kHz歌声変換モデルを更新
- 2024-10-07:
    - 音声コンテンツエンコーダをOpenAI Whisperに変更したv0.3事前学習済みモデルを更新
    - v0.3事前学習済みモデルの客観的評価結果を追加
- 2024-09-22:
    - NVIDIAのBigVGANを使用する歌声変換モデルを更新し、高音域の歌声を大幅に改善
    - Web UIで長い音声ファイルのチャンキングとストリーミング出力をサポート
- 2024-09-18:
    - 歌声変換用のf0条件付きモデルを更新
- 2024-09-14:
    - 同じ品質を達成するためのサイズ縮小と拡散ステップ数の削減、およびプロソディ保持の制御能力を追加したv0.2事前学習済みモデルを更新
    - コマンドライン推論スクリプトを追加
    - インストールと使用方法の説明を追加

## 謝辞🙏
- [Amphion](https://github.com/open-mmlab/Amphion) for providing computational resources and inspiration!
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for foundationing the real-time voice conversion
- [SEED-TTS](https://arxiv.org/abs/2406.02430) for the initial idea