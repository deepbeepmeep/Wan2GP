CONFIG_KEYS = ("configs", "configs2", "configs3")
CONFIG_NAME_KEY = "_name"


def get_config_items(configs):
    return [(config_id, config_def) for config_id, config_def in configs.items() if config_id != CONFIG_NAME_KEY]


def split_config_selection(selection):
    values = str(selection or "").split(",")
    return (values + [""] * len(CONFIG_KEYS))[:len(CONFIG_KEYS)]


def serialize_config_selection(*values):
    return ",".join(str(value or "") for value in values[:len(CONFIG_KEYS)]).rstrip(",")


def normalize_config_selection(config_groups, selection):
    values = split_config_selection(selection)
    for index, config_id in enumerate(values):
        if config_id not in config_groups[index] or config_id == CONFIG_NAME_KEY:
            values[index] = ""
    return serialize_config_selection(*values)


def selected_model_configs(config_groups, selection):
    for group, (configs, config_id) in enumerate(zip(config_groups, split_config_selection(selection)), 1):
        if not config_id:
            continue
        config_def = None if config_id == CONFIG_NAME_KEY else configs.get(config_id)
        if config_def is None:
            raise ValueError(f"Config '{config_id}' is not defined in Model Definition file")
        yield group, config_id, config_def


def get_config_name(configs):
    return configs.get(CONFIG_NAME_KEY) or "config"
