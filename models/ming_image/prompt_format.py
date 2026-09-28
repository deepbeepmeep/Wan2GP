"""Prompt cleanup for Ming's raw JSON and plain-language input."""


def strip_comment_lines(prompt: str) -> str:
    """Remove whole-line # comments while preserving inline text and hex colors."""
    return "\n".join(
        line for line in str(prompt).splitlines()
        if not line.lstrip().startswith("#")
    ).strip()
