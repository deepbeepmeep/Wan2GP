"""Runtime monkey patch that injects a Gradio background-scheduler fix into templates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH = True
GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE = False

_PATCH_SENTINEL = "window.__gradioFocusQueuePatch"
_TARGET_TEMPLATES = {"frontend/index.html", "frontend/share.html"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_javascript() -> str:
    if not ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH:
        return ""
    verbose = "true" if GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE else "false"
    return f"""
(function () {{
  if (typeof window === "undefined" || window.__gradioFocusQueuePatch) {{
    return;
  }}

  const nativeRequestAnimationFrame = window.requestAnimationFrame.bind(window);
  const nativeCancelAnimationFrame = window.cancelAnimationFrame.bind(window);
  const channel = typeof MessageChannel === "function" ? new MessageChannel() : null;

  function isElementVisible(element) {{
    if (!element) {{
      return false;
    }}
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }}

  function getVideoGenTab() {{
    return Array.from(document.querySelectorAll('[role="tab"]')).find((element) => element.textContent.trim() === "Video Generator") || null;
  }}

  function getVideoGenPanel() {{
    const tab = getVideoGenTab();
    const panelId = tab?.getAttribute("aria-controls");
    const panel = panelId ? document.getElementById(panelId) : null;
    return isElementVisible(panel) ? panel : null;
  }}

  function isVideoGenActive() {{
    return getVideoGenPanel() !== null;
  }}

  const patch = {{
    enabled: true,
    forceBackground: false,
    verbose: {verbose},
    nextId: 1,
    pending: new Map(),
    queue: [],
    nativeRequestAnimationFrame,
    nativeCancelAnimationFrame,
    isBackground() {{
      if (this.forceBackground) {{
        return true;
      }}
      try {{
        return document.visibilityState !== "visible" || !document.hasFocus();
      }} catch (_error) {{
        return false;
      }}
    }},
    shouldPatch() {{
      return this.enabled && this.isBackground() && isVideoGenActive();
    }},
    shouldPatchAnimationFrame(stack, callback) {{
      if (!this.shouldPatch()) {{
        return false;
      }}
      const stackText = String(stack || "");
      const source = String(callback || "");
      const isBlocksDispatch = stackText.includes("/assets/Blocks-") && source.includes("Jt(");
      const isCoreFlush = stackText.includes("/assets/index-") && source.includes("l.update(") && source.includes("ge.length") && source.includes("f.props[v.prop]=j");
      return isBlocksDispatch || isCoreFlush;
    }}
  }};

  function cancelSynthetic(id) {{
    const job = patch.pending.get(id);
    if (!job) {{
      return false;
    }}
    job.canceled = true;
    patch.pending.delete(id);
    return true;
  }}

  function dispatchSynthetic(job) {{
    if (!job || job.canceled) {{
      return;
    }}
    patch.pending.delete(job.id);
    try {{
      if (job.kind === "raf") {{
        job.callback(window.performance.now());
      }}
    }} catch (error) {{
      window.setTimeout(() => {{
        throw error;
      }}, 0);
    }}
  }}

  function flushOne() {{
    const job = patch.queue.shift();
    dispatchSynthetic(job);
  }}

  function scheduleSyntheticAnimationFrame(callback) {{
    if (!channel) {{
      return nativeRequestAnimationFrame(callback);
    }}
    const id = -patch.nextId++;
    const job = {{ id, kind: "raf", callback, args: null, canceled: false }};
    patch.pending.set(id, job);
    patch.queue.push(job);
    channel.port2.postMessage(id);
    if (patch.verbose) {{
      console.debug("[Gradio] focus queue synthetic animation frame", id);
    }}
    return id;
  }}

  if (channel) {{
    channel.port1.onmessage = flushOne;
  }}

  window.__gradioFocusQueuePatch = patch;

  window.requestAnimationFrame = function (callback) {{
    if (typeof callback !== "function") {{
      return nativeRequestAnimationFrame(callback);
    }}
    if (!patch.shouldPatch()) {{
      return nativeRequestAnimationFrame(callback);
    }}
    const stack = new Error().stack || "";
    if (!patch.shouldPatchAnimationFrame(stack, callback)) {{
      return nativeRequestAnimationFrame(callback);
    }}
    return scheduleSyntheticAnimationFrame(callback);
  }};

  window.cancelAnimationFrame = function (id) {{
    if (cancelSynthetic(id)) {{
      return;
    }}
    return nativeCancelAnimationFrame(id);
  }};

  console.info("[Gradio] focus queue patch installed");
}})();
"""


def _inject_script(template_source: str) -> str:
    if _PATCH_SENTINEL in template_source:
        return template_source
    script_tag = f"\n\t\t<script>\n{get_javascript()}\n\t\t</script>\n"
    module_tag = '<script type="module"'
    insert_at = template_source.find(module_tag)
    if insert_at != -1:
        return template_source[:insert_at] + script_tag + template_source[insert_at:]
    head_close = template_source.find("</head>")
    if head_close != -1:
        return template_source[:head_close] + script_tag + template_source[head_close:]
    return template_source + script_tag


def install() -> bool:
    if not ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH:
        return False
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv and sys.argv[0] else ""
    cwd = Path.cwd().resolve()
    if cwd != _PROJECT_ROOT and _PROJECT_ROOT not in cwd.parents and argv0 != "wgp.py":
        return False
    import gradio.routes as gradio_routes

    templates = getattr(gradio_routes, "templates", None)
    loader = getattr(getattr(templates, "env", None), "loader", None)
    if loader is None:
        return False
    if getattr(loader, "_focus_queue_patch_installed", False):
        return True

    original_get_source: Callable = loader.get_source

    def patched_get_source(environment, template):
        source, filename, uptodate = original_get_source(environment, template)
        if template in _TARGET_TEMPLATES:
            source = _inject_script(source)
        return source, filename, uptodate

    loader.get_source = patched_get_source
    loader._focus_queue_patch_installed = True
    loader._focus_queue_patch_original_get_source = original_get_source
    templates.env.cache.clear()
    return True


__all__ = [
    "ENABLE_GRADIO_FOCUS_QUEUE_MONKEYPATCH",
    "GRADIO_FOCUS_QUEUE_MONKEYPATCH_VERBOSE",
    "get_javascript",
    "install",
]
