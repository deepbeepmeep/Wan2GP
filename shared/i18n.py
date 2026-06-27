import json
import os
from typing import Callable

_LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locales")

_active_code = "en"
_active_dict: dict = {}
_cache: dict = {}
_DEFAULT_LANGUAGE = "en"


def _read_translations(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if not key.startswith("_") and isinstance(value, str)}


def list_languages() -> "list[tuple[str, str]]":
    found: "list[tuple[str, str]]" = []
    if os.path.isdir(_LOCALES_DIR):
        for name in sorted(os.listdir(_LOCALES_DIR)):
            if not name.endswith(".json"):
                continue
            code = name[:-5]
            display_name = code
            try:
                with open(os.path.join(_LOCALES_DIR, name), "r", encoding="utf-8") as reader:
                    data = json.load(reader)
                if isinstance(data, dict):
                    display_name = data.get("_language", code)
            except Exception:
                display_name = code
            found.append((display_name, code))
    if not any(code == _DEFAULT_LANGUAGE for _, code in found):
        found.insert(0, ("English", _DEFAULT_LANGUAGE))
    return found


def set_language(code: str) -> None:
    global _active_code, _active_dict
    code = (code or _DEFAULT_LANGUAGE).strip() or _DEFAULT_LANGUAGE
    if code in _cache:
        _active_code = code
        _active_dict = _cache[code]
        return
    candidates = [code]
    base = code.split("-", 1)[0]
    if base != code:
        candidates.append(base)
    for display_name, locale_code in list_languages():
        if code == display_name and locale_code not in candidates:
            candidates.append(locale_code)
    if _DEFAULT_LANGUAGE not in candidates:
        candidates.append(_DEFAULT_LANGUAGE)
    for candidate in candidates:
        path = os.path.join(_LOCALES_DIR, candidate + ".json")
        if os.path.isfile(path):
            try:
                translations = _read_translations(path)
            except Exception as exc:
                print(f"[i18n] Failed to load locale '{candidate}': {exc}")
                continue
            _cache[candidate] = translations
            _active_code = candidate
            _active_dict = translations
            if candidate != code:
                print(f"[i18n] Locale '{code}' not found, using '{candidate}' instead.")
            return
    _active_code = code
    _active_dict = {}
    if code != _DEFAULT_LANGUAGE:
        print(f"[i18n] No locale file for '{code}', falling back to English literals.")


def current_language() -> str:
    return _active_code


def tr(text: str, **kwargs) -> str:
    if not isinstance(text, str):
        return text
    result = _active_dict.get(text, text)
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            try:
                return text.format(**kwargs)
            except Exception:
                return text
    return result


def get_i18n_callable() -> Callable[[str], str]:
    return tr
