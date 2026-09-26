"""YuE2 composition-source flags and compatibility with saved cover settings."""


def composition_source(flags, has_score=False, extend=False):
    # A retains the shared audio upload/cleanup path; Q is an ABC file,
    # E requests planner continuation, and S explicitly ignores retained uploads.
    flags = flags or ""
    if not any(flag in flags for flag in "SAQ"):
        flags += "Q" if has_score else "S"
    if extend and "S" not in flags and "E" not in flags:
        flags += "E"
    return flags
