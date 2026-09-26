"""YuE2 composition-source flags and compatibility with saved cover settings."""


def composition_source(flags, has_score=False, extend=False):
    # A retains the shared audio upload/cleanup path; 1 is an ABC file,
    # 2 requests planner continuation, and 0 explicitly ignores retained uploads.
    flags = flags or ""
    if not any(flag in flags for flag in "0A1"):
        flags += "1" if has_score else "0"
    if extend and "0" not in flags and "2" not in flags:
        flags += "2"
    return flags
