SYSTEM_CONFIG_KEYS = ("system_configs", "system_configs2", "system_configs3")
USER_CONFIG_KEY = "configs"
CONFIG_KEYS = SYSTEM_CONFIG_KEYS + (USER_CONFIG_KEY,)
CONFIG_NAME_KEY = "_name"
CONFIG_DEFAULT_LABEL_KEY = "_default_label"
CONFIG_METADATA_KEYS = {CONFIG_NAME_KEY, CONFIG_DEFAULT_LABEL_KEY}


def get_config_items(configs):
    return [(config_id, config_def) for config_id, config_def in configs.items() if config_id not in CONFIG_METADATA_KEYS]


def split_config_selection(selection):
    values = str(selection or "").split(",")
    return (values + [""] * len(CONFIG_KEYS))[:len(CONFIG_KEYS)]


def serialize_config_selection(*values):
    return ",".join(str(value or "") for value in values[:len(CONFIG_KEYS)]).rstrip(",")


def normalize_config_selection(config_groups, selection):
    values = split_config_selection(selection)
    for index, config_id in enumerate(values):
        if config_id not in config_groups[index] or config_id in CONFIG_METADATA_KEYS:
            values[index] = ""
    return serialize_config_selection(*values)


def selected_model_configs(config_groups, selection):
    for group, (configs, config_id) in enumerate(zip(config_groups, split_config_selection(selection)), 1):
        if not config_id:
            continue
        config_def = None if config_id in CONFIG_METADATA_KEYS else configs.get(config_id)
        if config_def is None:
            raise ValueError(f"Config '{config_id}' is not defined in Model Definition file")
        yield group, config_id, config_def


def format_config_selection(config_groups, selection):
    summaries = []
    for config_key, configs, config_id in zip(CONFIG_KEYS, config_groups, split_config_selection(selection)):
        if not config_id:
            continue
        config_def = None if config_id in CONFIG_METADATA_KEYS else configs.get(config_id)
        config_label = configs.get(CONFIG_NAME_KEY) or config_key
        choice_label = (config_def.get("name") or config_id) if config_def is not None else config_id
        summaries.append(f"{config_label}={choice_label}")
    return ", ".join(summaries)


def get_config_name(configs):
    return configs.get(CONFIG_NAME_KEY) or "config"


def get_default_label(configs):
    return configs.get(CONFIG_DEFAULT_LABEL_KEY, "Default")
