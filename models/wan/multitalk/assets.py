"""Download declaration shared by all consumers of these weights."""


def query_download_def(include_readme=False):
    definition = {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": ['chinese-wav2vec2-base'], "fileList": [['config.json', 'pytorch_model.bin', 'preprocessor_config.json']]}
    if include_readme:
        definition["fileList"][0].append("readme.txt")
    return definition
