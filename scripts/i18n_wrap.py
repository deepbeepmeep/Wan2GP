"""One-off developer tool: wrap user-facing strings with i18n.tr(...).

Wraps:
  * gr.Info / gr.Warning / gr.Error first string/f-string argument.
  * label= / info= / placeholder= / title= keyword string/f-string values.

It is AST-based so multi-line calls and f-strings are handled precisely. Complex
f-strings (fields that are not plain names) are skipped and reported for manual
review. Files that are changed get a ``from shared import i18n`` import added if
absent. Not imported at runtime.

Usage::

    python scripts/i18n_wrap.py toasts   [--apply]
    python scripts/i18n_wrap.py labels   [--apply]
    python scripts/i18n_wrap.py all      [--apply]
"""

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOAST_FUNCS = {"Info", "Warning", "Error"}
LABEL_KEYWORDS = {"label", "info", "placeholder", "title"}
COMPONENT_FUNCS = {"Tab", "Button", "Accordion"}

TOAST_DIRS = ["wgp.py", "shared", "plugins", "postprocessing", "preprocessing", "models"]
LABEL_DIRS = ["wgp.py", "shared/gradio", "plugins"]


def iter_files(dirs):
    seen = set()
    for entry in dirs:
        path = os.path.join(ROOT, entry.replace("/", os.sep))
        if os.path.isfile(path):
            if path not in seen:
                seen.add(path)
                yield path
        elif os.path.isdir(path):
            for dp, _, fs in os.walk(path):
                for f in fs:
                    if f.endswith(".py"):
                        p = os.path.join(dp, f)
                        if p not in seen:
                            seen.add(p)
                            yield p


def line_offset_table(source):
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def span(source, node):
    offsets = line_offset_table(source)
    lines = source.split("\n")

    def byte_to_char(line_index, byte_col):
        line_text = lines[line_index]
        prefix = line_text.encode("utf-8")[:byte_col]
        return len(prefix.decode("utf-8", "replace"))

    start_char = byte_to_char(node.lineno - 1, node.col_offset)
    end_char = byte_to_char(node.end_lineno - 1, node.end_col_offset)
    return offsets[node.lineno - 1] + start_char, offsets[node.end_lineno - 1] + end_char


def py_str_literal(text):
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def format_spec_text(node):
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def build_translation(node, source):
    """Return (replacement_text, key) or None if not wrappable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
        return "i18n.tr(" + py_str_literal(text) + ")", text
    if isinstance(node, ast.JoinedStr):
        parts = []
        kwargs = []
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
                conv = ""
                if value.conversion == 114:
                    conv = "!r"
                elif value.conversion == 115:
                    conv = "!s"
                elif value.conversion == 97:
                    conv = "!a"
                spec = ""
                if value.format_spec is not None:
                    spec_text = format_spec_text(value.format_spec)
                    spec = ":" + spec_text if spec_text else ""
                parts.append("{" + name + conv + spec + "}")
                kwargs.append(name)
            else:
                ok = False
                break
        if not ok:
            return None
        template = "".join(parts)
        call = "i18n.tr(" + py_str_literal(template)
        if kwargs:
            call += ", " + ", ".join(name + "=" + name for name in kwargs)
        call += ")"
        return call, template
    return None


def is_gr_toast(func):
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "gr"
        and func.attr in TOAST_FUNCS
    )


def is_gr_component(func):
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "gr"
        and func.attr in COMPONENT_FUNCS
    )


def ensure_import(source):
    if "from shared import i18n" in source or "from shared.i18n" in source:
        return source, False
    lines = source.split("\n")
    anchor = None
    for i, line in enumerate(lines):
        if line.startswith("import gradio as gr"):
            anchor = i
            break
    if anchor is None:
        for i, line in enumerate(lines):
            if line.startswith("from shared import") or line.startswith("from shared."):
                anchor = i
                break
    if anchor is None:
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                anchor = i
                break
    insert_at = (anchor + 1) if anchor is not None else 0
    lines.insert(insert_at, "from shared import i18n")
    return "\n".join(lines), True


def transform(path, mode):
    try:
        with open(path, "r", encoding="utf-8") as reader:
            source = reader.read()
    except Exception:
        return None
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return None

    edits = []
    keys = set()
    skipped = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        candidates = []
        if mode in ("toasts", "all") and is_gr_toast(node.func) and node.args:
            candidates.append(node.args[0])
        if mode in ("components", "all") and is_gr_component(node.func) and node.args:
            candidates.append(node.args[0])
        if mode in ("labels", "components", "all"):
            for kw in node.keywords:
                if kw.arg in LABEL_KEYWORDS:
                    candidates.append(kw.value)
        for target in candidates:
            try:
                start, end = span(source, target)
            except Exception:
                continue
            prefix = source[max(0, start - 10):start]
            if prefix.endswith("tr(") or prefix.endswith(".tr("):
                continue
            result = build_translation(target, source)
            if result is None:
                skipped += 1
                continue
            replacement, key = result
            edits.append((start, end, replacement))
            keys.add(key)

    if not edits:
        return None
    edits.sort(key=lambda item: item[0], reverse=True)
    new_source = source
    for start, end, replacement in edits:
        new_source = new_source[:start] + replacement + new_source[end:]
    new_source, _ = ensure_import(new_source)
    return new_source, len(edits), skipped, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["toasts", "labels", "components", "all"])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dirs = TOAST_DIRS if args.mode == "toasts" else LABEL_DIRS if args.mode in ("labels", "components") else sorted(set(TOAST_DIRS) | set(LABEL_DIRS))
    total_edits = 0
    total_skipped = 0
    changed_files = 0
    all_keys = set()
    for path in iter_files(dirs):
        result = transform(path, args.mode)
        if result is None:
            continue
        new_source, edits, skipped, keys = result
        changed_files += 1
        total_edits += edits
        total_skipped += skipped
        all_keys |= keys
        rel = os.path.relpath(path, ROOT)
        if args.apply:
            with open(path, "w", encoding="utf-8") as writer:
                writer.write(new_source)
        print(f"  {rel}: {edits} wrap(s), {skipped} skipped")

    action = "applied" if args.apply else "DRY-RUN"
    print(f"\n[{action}] mode={args.mode} files={changed_files} wraps={total_edits} skipped={total_skipped} unique_keys={len(all_keys)}")


if __name__ == "__main__":
    sys.exit(main())
