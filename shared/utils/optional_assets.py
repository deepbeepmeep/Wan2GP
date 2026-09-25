"""Optional asset inventory for the model manager; never a generation prerequisite."""


def query_download_defs(server_config=None):
    from preprocessing.processors import query_preprocessor_download_def
    from preprocessing.matanyone.utils.model_assets import query_matanyone_download_def
    from preprocessing.speaker_separator.assets import query_speaker_separator_download_def
    from preprocessing.roformer.assets import query_download_def as roformer
    from models.hyvideo.data_kits.assets import query_download_def as face
    from models.wan.fantasytalking.assets import query_download_def as wav2vec
    from models.wan.multitalk.assets import query_download_def as chinese_wav2vec
    from postprocessing import spatial_upsamplers, temporal_upsamplers, audio_processors
    from shared.deepy.assets import query_deepy_download_defs
    from shared.prompt_enhancer.assets import query_prompt_enhancer_download_defs

    definitions = [query_preprocessor_download_def(kind, server_config) for kind in ("pose", "scribble", "flow", "depth")]
    definitions.extend([query_matanyone_download_def(server_config), roformer(), face(), wav2vec(), chinese_wav2vec()])
    definitions.extend(query_speaker_separator_download_def())
    for processors in (spatial_upsamplers, temporal_upsamplers, audio_processors):
        definitions.extend(processors.query_download_defs(enabled_only=False))
    definitions.extend(query_prompt_enhancer_download_defs())
    definitions.extend(query_deepy_download_defs())
    return definitions
