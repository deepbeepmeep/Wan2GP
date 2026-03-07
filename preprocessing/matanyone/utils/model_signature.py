import os


MATANYONE2_SIGNATURE_OFFSET = 61576
MATANYONE2_SIGNATURE_BYTES = bytes([155, 244, 100, 61, 100, 179, 194, 60, 92, 239, 124, 186, 254, 127, 168, 61])


def delete_if_not_matanyone2_model(model_path):
    if model_path is None or not os.path.isfile(model_path):
        return False

    with open(model_path, "rb") as reader:
        reader.seek(MATANYONE2_SIGNATURE_OFFSET)
        actual = reader.read(len(MATANYONE2_SIGNATURE_BYTES))

    if actual == MATANYONE2_SIGNATURE_BYTES:
        return False

    os.remove(model_path)
    return True
