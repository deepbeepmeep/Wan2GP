"""Developer aid: scan the codebase for i18n candidate keys.

It reports and optionally writes a sorted, deduplicated candidate key list to
``scripts/i18n_audit.json`` (kept out of ``locales/`` so it does not pollute the
language dropdown). After wrapping, the canonical keys are the string literals
passed to ``i18n.tr(...)``. This script also lists still-unwrapped
``gr.Info/Warning/Error`` string literals and ``label=``/``info=`` literals so
drift can be spotted.

Not imported at runtime. Run with the project venv::

    python scripts/extract_i18n_strings.py            # print a report
    python scripts/extract_i18n_strings.py --write     # also write locales/en_strings.json
"""

import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["wgp.py", "shared", "plugins", "postprocessing", "preprocessing", "models"]
TOAST_FUNCS = {"Info", "Warning", "Error"}
LABEL_KEYWORDS = {"label", "info", "placeholder", "title", "description"}


def iter_py_files():
    for entry in SCAN_DIRS:
        path = os.path.join(ROOT, entry)
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dp, _, fs in os.walk(path):
                for f in fs:
                    if f.endswith(".py"):
                        yield os.path.join(dp, f)


def const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def fstr_template(node):
    """Return (template, ok) for a JoinedStr using str.format placeholders."""
    if not isinstance(node, ast.JoinedStr):
        return None, False
    parts = []
    fields = {}
    ok = True
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value.replace("{", "{{").replace("}", "}}"))
        elif isinstance(value, ast.FormattedValue):
            inner = value.value
            if not isinstance(inner, ast.Name):
                ok = False
                break
            name = inner.id
            fields[name] = True
            conv = ""
            if value.conversion == 114:
                conv = "!r"
            elif value.conversion == 115:
                conv = "!s"
            elif value.conversion == 97:
                conv = "!a"
            spec = ""
            if value.format_spec is not None:
                spec_text = _format_spec_text(value.format_spec)
                spec = ":" + spec_text if spec_text else ""
            parts.append("{" + name + conv + spec + "}")
        else:
            ok = False
            break
    if not ok:
        return None, False
    return "".join(parts), True


def _format_spec_text(node):
    if isinstance(node, ast.JoinedStr):
        out = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out.append(value.value)
        return "".join(out)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def is_gr_attr(node, names):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "gr"
        and node.attr in names
    )


def analyze(path):
    try:
        with open(path, "r", encoding="utf-8") as reader:
            source = reader.read()
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return None
    keys_tr = set()
    unwrapped_toasts = set()
    unwrapped_labels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if is_gr_attr(node.func, TOAST_FUNCS) and node.args:
                first = node.args[0]
                text = const_str(first)
                if text is not None:
                    tmpl, ok = fstr_template(first)
                    candidate = tmpl if ok and tmpl is not None else text
                    if _looks_wrapped(first, source):
                        keys_tr.add(candidate)
                    else:
                        unwrapped_toasts.add(candidate)
            for kw in node.keywords:
                if kw.arg in LABEL_KEYWORDS:
                    text = const_str(kw.value)
                    if text is not None and not _looks_wrapped(kw.value, source):
                        unwrapped_labels.add(text)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("tr",) and node.args:
            text = const_str(node.args[0])
            if text is not None:
                keys_tr.add(text)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tr" and node.args:
            text = const_str(node.args[0])
            if text is not None:
                keys_tr.add(text)
    return keys_tr, unwrapped_toasts, unwrapped_labels


def _looks_wrapped(node, source):
    if not hasattr(node, "col_offset") or node.col_offset is None:
        return False
    try:
        prefix = source.rsplit("\n", 1)[1][max(0, node.col_offset - 9):node.col_offset]
    except Exception:
        return False
    return prefix.endswith("tr(") or prefix.endswith(".tr(")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out", default=os.path.join(ROOT, "scripts", "i18n_audit.json"))
    args = parser.parse_args()

    all_keys = set()
    all_unwrapped_toasts = set()
    all_unwrapped_labels = set()
    files = 0
    for path in iter_py_files():
        result = analyze(path)
        if result is None:
            continue
        files += 1
        keys, toasts, labels = result
        all_keys |= keys
        all_unwrapped_toasts |= toasts
        all_unwrapped_labels |= labels

    print(f"Scanned {files} python files.")
    print(f"Wrapped keys (i18n.tr / tr): {len(all_keys)}")
    print(f"Unwrapped gr.Info/Warning/Error string literals: {len(all_unwrapped_toasts)}")
    print(f"Unwrapped label/info/placeholder/title literals: {len(all_unwrapped_labels)}")

    if args.write:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "_language": "English",
            "_keys": sorted(all_keys),
            "_unwrapped_toasts": sorted(all_unwrapped_toasts),
            "_unwrapped_labels": sorted(all_unwrapped_labels),
        }
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
