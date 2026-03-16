import os


MATANYONE_V1 = "v1"
MATANYONE_V2 = "v2"
MATANYONE2_SIGNATURE_OFFSET = 61576
MATANYONE1_SIGNATURE_BYTES = bytes([23, 65, 57, 61, 52, 154, 201, 60, 231, 125, 180, 188, 187, 85, 166, 61])
MATANYONE2_SIGNATURE_BYTES = bytes([155, 244, 100, 61, 100, 179, 194, 60, 92, 239, 124, 186, 254, 127, 168, 61])


def detect_matanyone_model_version(model_path):
    if model_path is None or not os.path.isfile(model_path):
        return None

    with open(model_path, "rb") as reader:
        reader.seek(MATANYONE2_SIGNATURE_OFFSET)
        actual = reader.read(len(MATANYONE2_SIGNATURE_BYTES))

    if actual == MATANYONE1_SIGNATURE_BYTES:
        return MATANYONE_V1
    if actual == MATANYONE2_SIGNATURE_BYTES:
        return MATANYONE_V2
    return None


def delete_if_not_matanyone2_model(model_path):
    if detect_matanyone_model_version(model_path) is not None:
        return False

    os.remove(model_path)
    return True
