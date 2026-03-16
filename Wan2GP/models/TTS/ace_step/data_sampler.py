import json
from pathlib import Path
import random


DEFAULT_ROOT_DIR = "examples/default/input_params"
ZH_RAP_LORA_ROOT_DIR = "examples/zh_rap_lora/input_params"

class DataSampler:
    def __init__(self, root_dir=DEFAULT_ROOT_DIR):
        self.root_dir = root_dir
        self.input_params_files = list(Path(self.root_dir).glob("*.json"))
        self.zh_rap_lora_input_params_files = list(Path(ZH_RAP_LORA_ROOT_DIR).glob("*.json"))
        self.zh_rap_lora_input_params_files += list(Path(ZH_RAP_LORA_ROOT_DIR).glob("*.json"))

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def sample(self, lora_name_or_path=None):
        if lora_name_or_path is None or lora_name_or_path == "none":
            json_path = random.choice(self.input_params_files)
            json_data = self.load_json(json_path)
        else:
            json_path = random.choice(self.zh_rap_lora_input_params_files)
            json_data = self.load_json(json_path)
            # Update the lora_name in the json_data
            json_data["lora_name_or_path"] = lora_name_or_path

        return json_data
