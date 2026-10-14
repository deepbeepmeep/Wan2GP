# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from markdown.extensions import Extension


class EscapeHtmlExtension(Extension):
    """Keep raw HTML as text without pre-escaping Markdown entities or code."""

    def extendMarkdown(self, md):
        # Markdown's serializer escapes text; code processors escape code once.
        # Keep the entity processor so character references still work in prose.
        md.preprocessors.deregister("html_block")
        md.inlinePatterns.deregister("html")
