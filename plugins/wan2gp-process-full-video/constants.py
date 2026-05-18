from __future__ import annotations

RATIO_CHOICES = [("1:1", "1:1"), ("4:3", "4:3"), ("3:4", "3:4"), ("16:9", "16:9"), ("9:16", "9:16"), ("21:9", "21:9"), ("9:21", "9:21")]
RATIO_CHOICES_WITH_EMPTY = [("", "")] + RATIO_CHOICES
DEFAULT_SOURCE_PATH = ""
DEFAULT_OUTPUT_PATH = ""
ADD_USER_SETTINGS_MODEL_TYPE = "__add_user_settings__"
ADD_USER_SETTINGS_LABEL = "<Add User Settings>"
NO_USER_SETTINGS_VALUE = "__no_user_settings__"
NO_USER_SETTINGS_LABEL = "<No choice>"
USER_SETTINGS_HINT_HTML = "<div style='font-size:10px;line-height:1;opacity:.65;'>* user settings</div>"
MAX_STATUS_REFRESH_HZ = 3.0
STATUS_REFRESH_INTERVAL_SECONDS = 1.0 / MAX_STATUS_REFRESH_HZ
SUPPORTED_OUTPUT_CONTAINERS = {"mkv", "mp4"}
