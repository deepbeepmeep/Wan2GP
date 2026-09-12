"""Selection metadata and relevance ordering for MCP v2 model discovery."""

import copy
import json
import re
from functools import lru_cache
from pathlib import Path


PREFERENCE_CHOICES = {"speed": ("fast", "standard", "any"), "size": ("smaller", "larger", "any")}


@lru_cache(maxsize=256)
def _read_profile(path, modified, size):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    priority = content.get("profile_priority", 0)
    if type(priority) is not int:
        raise ValueError(f"profile_priority must be an integer: {path}")
    return content


def read_profile(path):
    path = Path(path)
    stat = path.stat()
    return copy.deepcopy(_read_profile(str(path.resolve()), stat.st_mtime_ns, stat.st_size))


def prioritize_profiles(entries):
    accelerators = [entry for entry in entries if entry["type"] == "accelerator profile"]
    for entry in accelerators:
        entry["profile_priority"] = read_profile(entry["_path"]).get("profile_priority", 0)
    accelerators.sort(key=lambda entry: (-entry["profile_priority"], entry["id"].casefold()))
    if accelerators and accelerators[0]["profile_priority"] > 0 and (len(accelerators) == 1 or accelerators[0]["profile_priority"] > accelerators[1]["profile_priority"]):
        accelerators[0]["recommended"] = True
    return accelerators + [entry for entry in entries if entry["type"] != "accelerator profile"]


def normalize_preferences(values):
    if not isinstance(values, dict) or values.keys() - PREFERENCE_CHOICES.keys():
        raise ValueError("preferences accepts speed and size only")
    result = {"speed": "any", "size": "any", **values}
    for key, value in result.items():
        if value not in PREFERENCE_CHOICES[key]:
            raise ValueError(f"Invalid {key} preference: {value}")
    return result


def _normalized(text):
    return " ".join(text.casefold().split())


def _words(text):
    return set(re.findall(r"\w+", text.casefold()))


def speciality_catalog(records):
    catalog = {}
    for record in records:
        for speciality in record.get("specialities", []):
            name = _normalized(speciality["name"])
            item = catalog.setdefault(name, {"name": name, "aliases": set(), "descriptions": set()})
            item["aliases"].update(speciality.get("aliases", []))
            if speciality.get("description"):
                item["descriptions"].add(speciality["description"])
    return [{"name": name, **({"aliases": sorted(item["aliases"])} if item["aliases"] else {}), **({"description": " ".join(sorted(item["descriptions"]))} if item["descriptions"] else {})} for name, item in sorted(catalog.items())]


def rank_models(records, specialities, preferences):
    if not isinstance(specialities, list) or any(not isinstance(term, str) or not term.strip() for term in specialities):
        raise ValueError("specialities must be a list of non-empty strings")
    terms = list(dict.fromkeys(_normalized(term) for term in specialities))
    ranked = []
    for record in records:
        names, words = set(), set()
        for speciality in record.get("specialities", []):
            names.update(_normalized(value) for value in [speciality["name"], *speciality.get("aliases", [])])
            words.update(_words(" ".join([speciality["name"], *speciality.get("aliases", []), speciality.get("description", "")])))
        matched = [term for term in terms if term in names]
        word_matches = {term: sorted(_words(term) & words) for term in terms if term not in names and _words(term) & words}
        if terms and not matched and not word_matches:
            continue
        speed = preferences["speed"]
        score = int(speed == "fast" and record["accelerated"] in ("native", "profiles") or speed == "standard" and record["accelerated"] in ("none", "profiles"))
        score += int(preferences["size"] != "any" and record.get("size") == {"smaller": "lighter", "larger": "large"}.get(preferences["size"]))
        ranked.append((len(matched), len(word_matches), score, record, matched, word_matches))
    exact = any(item[0] == len(terms) for item in ranked)
    if exact:
        ranked = [item for item in ranked if item[0] == len(terms)]
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]["name"].casefold(), item[3]["model_type"]))
    result = []
    for _, _, _, record, matched, word_matches in ranked:
        item = dict(record)
        if not exact:
            item.update(matched_specialities=matched, unmatched_specialities=[term for term in terms if term not in matched])
            if word_matches:
                item["word_matches"] = word_matches
        result.append(item)
    return result, "exact" if exact else "partial" if result else "none"
