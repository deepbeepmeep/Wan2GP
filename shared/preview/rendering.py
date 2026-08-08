from __future__ import annotations

import base64
import html

from .types import PreviewMedia


def preview_media_to_html(media: PreviewMedia, *, height: int = 200) -> str:
    if not media or not media.data:
        return ""
    uri = "data:%s;base64,%s" % (media.mime_type, base64.b64encode(media.data).decode("ascii"))
    labels = [f"Tiny VAE · Step {media.step}/{media.total_steps} · {media.frame_count} preview frames"]
    if media.pass_no is not None:
        labels.append(f"Pass {media.pass_no}")
    if media.window_no is not None:
        labels.append(f"Window {media.window_no}")
    if media.warning:
        labels.append(str(media.warning))
    label = html.escape(" · ".join(labels))
    if media.media_kind == "video":
        generation = html.escape(str(media.generation_id), quote=True)
        ontimeupdate = "const d=this.duration;if(Number.isFinite(d)&&d>0){(window.__wangpPreviewPlayback||(window.__wangpPreviewPlayback={}))[this.dataset.previewGeneration]=Math.min(Math.max(0,this.currentTime/d),1);}"
        onloadedmetadata = "const s=window.__wangpPreviewPlayback&&window.__wangpPreviewPlayback[this.dataset.previewGeneration],d=this.duration;if(s!=null&&Number.isFinite(d)&&d>0){this.currentTime=Math.min(Math.max(0,s*d),Math.max(0,d-0.001));}"
        media_html = f'<video src="{uri}" data-preview-generation="{generation}" ontimeupdate="{ontimeupdate}" onloadedmetadata="{onloadedmetadata}" autoplay loop muted playsinline style="max-height:100%;max-width:100%;object-fit:contain"></video>'
    else:
        media_html = f'<img src="{uri}" style="max-height:100%;max-width:100%;object-fit:contain" alt="Live Tiny VAE generation preview">'
    if media.media_kind == "video":
        return f'<div style="display:flex;flex-direction:column;justify-content:center;align-items:center;height:{int(height)}px">{media_html}<small>{label}</small></div>'
    return f'<div style="display:flex;flex-direction:column;justify-content:center;align-items:center;height:{int(height)}px;cursor:pointer" onclick="showImageModal(\'preview_0\')">{media_html}<small>{label}</small></div>'
