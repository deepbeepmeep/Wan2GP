from __future__ import annotations
import os, io, tempfile, mimetypes
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import gradio as gr
import PIL
from PIL import Image as PILImage

FilePath = str
ImageLike = Union["PIL.Image.Image", Any]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ogv"}

def get_state(state):
    return state if isinstance(state, dict) else state.value

def get_list( objs):
    if objs is None:
        return []
    return [ obj[0] if isinstance(obj, tuple) else obj for obj in objs]

class AdvancedMediaGallery:
    def __init__(
        self,
        label: str = "Media",
        *,
        media_mode: Literal["image", "video"] = "image",
        height = None,
        columns: Union[int, Tuple[int, ...]] = 6,
        show_label: bool = True,
        initial: Optional[Sequence[Union[FilePath, ImageLike]]] = None,
        elem_id: Optional[str] = None,
        elem_classes: Optional[Sequence[str]] = ("adv-media-gallery",),
        accept_filter: bool = True,        # restrict Add-button dialog to allowed extensions
        single_image_mode: bool = False,   # start in single-image mode (Add replaces)
    ):
        assert media_mode in ("image", "video")
        self.label = label
        self.media_mode = media_mode
        self.height = height
        self.columns = columns
        self.show_label = show_label
        self.elem_id = elem_id
        self.elem_classes = list(elem_classes) if elem_classes else None
        self.accept_filter = accept_filter

        items = self._normalize_initial(initial or [], media_mode)

        # Components (filled on mount)
        self.container: Optional[gr.Column] = None
        self.gallery: Optional[gr.Gallery] = None
        self.upload_btn: Optional[gr.UploadButton] = None
        self.btn_remove: Optional[gr.Button] = None
        self.btn_left: Optional[gr.Button] = None
        self.btn_right: Optional[gr.Button] = None
        self.btn_clear: Optional[gr.Button] = None

        # Single dict state
        self.state: Optional[gr.State] = None
        self._initial_state: Dict[str, Any] = {
            "items": items,
            "selected": (len(items) - 1) if items else None,
            "single": bool(single_image_mode),
            "mode": self.media_mode,
        }

    # ---------------- helpers ----------------

    def _normalize_initial(self, items: Sequence[Union[FilePath, ImageLike]], mode: str) -> List[Any]:
        out: List[Any] = []
        if mode == "image":
            for it in items:
                p = self._ensure_image_item(it)
                if p is not None:
                    out.append(p)
        else:
            for it in items:
                if isinstance(item, tuple): item = item[0]
                if isinstance(it, str) and self._is_video_path(it):
                    out.append(os.path.abspath(it))
        return out

    def _ensure_image_item(self, item: Union[FilePath, ImageLike]) -> Optional[Any]:
        # Accept a path to an image, or a PIL.Image/np.ndarray -> save temp PNG and return its path
        if isinstance(item, tuple): item = item[0]
        if isinstance(item, str):
            return os.path.abspath(item) if self._is_image_path(item) else None
        if PILImage is None:
            return None
        try:
            if isinstance(item, PILImage.Image):
                img = item
            else:
                import numpy as np  # type: ignore
                if isinstance(item, np.ndarray):
                    img = PILImage.fromarray(item)
                elif hasattr(item, "read"):
                    data = item.read()
                    img = PILImage.open(io.BytesIO(data)).convert("RGBA")
                else:
                    return None
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            return tmp.name
        except Exception:
            return None

    @staticmethod
    def _extract_path(obj: Any) -> Optional[str]:
        # Try to get a filesystem path (for mode filtering); otherwise None.
        if isinstance(obj, str):
            return obj
        try:
            import pathlib
            if isinstance(obj, pathlib.Path):  # type: ignore
                return str(obj)
        except Exception:
            pass
        if isinstance(obj, dict):
            return obj.get("path") or obj.get("name")
        for attr in ("path", "name"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if isinstance(val, str):
                        return val
                except Exception:
                    pass
        return None

    @staticmethod
    def _is_image_path(p: str) -> bool:
        ext = os.path.splitext(p)[1].lower()
        if ext in IMAGE_EXTS:
            return True
        mt, _ = mimetypes.guess_type(p)
        return bool(mt and mt.startswith("image/"))

    @staticmethod
    def _is_video_path(p: str) -> bool:
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXTS:
            return True
        mt, _ = mimetypes.guess_type(p)
        return bool(mt and mt.startswith("video/"))

    def _filter_items_by_mode(self, items: List[Any]) -> List[Any]:
        # Enforce image-only or video-only collection regardless of how files were added.
        out: List[Any] = []
        if self.media_mode == "image":
            for it in items:
                p = self._extract_path(it)
                if p is None:
                    # No path: likely an image object added programmatically => keep
                    out.append(it)
                elif self._is_image_path(p):
                    out.append(os.path.abspath(p))
        else:
            for it in items:
                p = self._extract_path(it)
                if p is not None and self._is_video_path(p):
                    out.append(os.path.abspath(p))
        return out

    @staticmethod
    def _concat_and_optionally_dedupe(cur: List[Any], add: List[Any]) -> List[Any]:
        # Keep it simple: dedupe by path when available, else allow duplicates.
        seen_paths = set()
        def key(x: Any) -> Optional[str]:
            if isinstance(x, str): return os.path.abspath(x)
            try:
                import pathlib
                if isinstance(x, pathlib.Path):  # type: ignore
                    return os.path.abspath(str(x))
            except Exception:
                pass
            if isinstance(x, dict):
                p = x.get("path") or x.get("name")
                return os.path.abspath(p) if isinstance(p, str) else None
            for attr in ("path", "name"):
                if hasattr(x, attr):
                    try:
                        v = getattr(x, attr)
                        return os.path.abspath(v) if isinstance(v, str) else None
                    except Exception:
                        pass
            return None

        out: List[Any] = []
        for lst in (cur, add):
            for it in lst:
                k = key(it)
                if k is None or k not in seen_paths:
                    out.append(it)
                    if k is not None:
                        seen_paths.add(k)
        return out

    @staticmethod
    def _paths_from_payload(payload: Any) -> List[Any]:
        # Return as raw objects (paths/dicts/UploadedFile) to feed Gallery directly.
        if payload is None:
            return []
        if isinstance(payload, (list, tuple, set)):
            return list(payload)
        return [payload]

    # ---------------- event handlers ----------------

    def _on_select(self, state: Dict[str, Any], gallery, evt: gr.SelectData) :
        # Mirror the selected index into state and the gallery (server-side selected_index)
        idx = None
        if evt is not None and hasattr(evt, "index"):
            ix = evt.index
            if isinstance(ix, int):
                idx = ix
            elif isinstance(ix, (tuple, list)) and ix and isinstance(ix[0], int):
                if isinstance(self.columns, int) and len(ix) >= 2:
                    idx = ix[0] * max(1, int(self.columns)) + ix[1]
                else:
                    idx = ix[0]
        st = get_state(state)
        n = len(get_list(gallery))
        sel = idx if (idx is not None and 0 <= idx < n) else None
        st["selected"] = sel
        # return gr.update(selected_index=sel), st
        # return gr.update(), st
        return st

    def _on_gallery_change(self, value: List[Any], state: Dict[str, Any]) :
        # Fires when users add/drag/drop/delete via the Gallery itself.
        items_filtered = self._filter_items_by_mode(list(value or []))
        st = get_state(state)
        st["items"] = items_filtered
        # Keep selection if still valid, else default to last
        old_sel = st.get("selected", None)
        if old_sel is None or not (0 <= old_sel < len(items_filtered)):
            new_sel = (len(items_filtered) - 1) if items_filtered else None
        else:
            new_sel = old_sel
        st["selected"] = new_sel
        # return gr.update(value=items_filtered, selected_index=new_sel), st
        # return gr.update(value=items_filtered), st

        return gr.update(), st

    def _on_add(self, files_payload: Any, state: Dict[str, Any], gallery):
        """
        Insert added items right AFTER the currently selected index.
        Keeps the same ordering as chosen in the file picker, dedupes by path,
        and re-selects the last inserted item.
        """
        # New items (respect image/video mode)
        new_items = self._filter_items_by_mode(self._paths_from_payload(files_payload))

        st = get_state(state)
        cur: List[Any] = get_list(gallery)
        sel = st.get("selected", None)
        if sel is None:
            sel = (len(cur) -1) if len(cur)>0 else 0
        single = bool(st.get("single", False))

        # Nothing to add: keep as-is
        if not new_items:
            return gr.update(value=cur, selected_index=st.get("selected")), st

        # Single-image mode: replace
        if single:
            st["items"] = [new_items[-1]]
            st["selected"] = 0
            return gr.update(value=st["items"], selected_index=0), st

        # ---------- helpers ----------
        def key_of(it: Any) -> Optional[str]:
            # Prefer class helper if present
            if hasattr(self, "_extract_path"):
                p = self._extract_path(it)  # type: ignore
            else:
                p = it if isinstance(it, str) else None
                if p is None and isinstance(it, dict):
                    p = it.get("path") or it.get("name")
                if p is None and hasattr(it, "path"):
                    try: p = getattr(it, "path")
                    except Exception: p = None
                if p is None and hasattr(it, "name"):
                    try: p = getattr(it, "name")
                    except Exception: p = None
            return os.path.abspath(p) if isinstance(p, str) else None

        # Dedupe the incoming batch by path, preserve order
        seen_new = set()
        incoming: List[Any] = []
        for it in new_items:
            k = key_of(it)
            if k is None or k not in seen_new:
                incoming.append(it)
                if k is not None:
                    seen_new.add(k)

        # Remove any existing occurrences of the incoming items from current list,
        # BUT keep the currently selected item even if it's also in incoming.
        cur_clean: List[Any] = []
        # sel_item = cur[sel] if (sel is not None and 0 <= sel < len(cur)) else None
        # for idx, it in enumerate(cur):
        #     k = key_of(it)
        #     if it is sel_item:
        #         cur_clean.append(it)
        #         continue
        #     if k is not None and k in seen_new:
        #         continue  # drop duplicate; we'll reinsert at the target spot
        #     cur_clean.append(it)

        # # Compute insertion position: right AFTER the (possibly shifted) selected item
        # if sel_item is not None:
        #     # find sel_item's new index in cur_clean
        #     try:
        #         pos_sel = cur_clean.index(sel_item)
        #     except ValueError:
        #         # Shouldn't happen, but fall back to end
        #         pos_sel = len(cur_clean) - 1
        #     insert_pos = pos_sel + 1
        # else:
        #     insert_pos = len(cur_clean)  # no selection -> append at end
        insert_pos = min(sel, len(cur) -1)
        cur_clean = cur
        # Build final list and selection
        merged = cur_clean[:insert_pos+1] + incoming + cur_clean[insert_pos+1:]
        new_sel = insert_pos + len(incoming)   # select the last inserted item

        st["items"] = merged
        st["selected"] = new_sel
        return gr.update(value=merged, selected_index=new_sel), st

    def _on_remove(self, state: Dict[str, Any], gallery) :
        st = get_state(state); items: List[Any] = get_list(gallery); sel = st.get("selected", None)
        if sel is None or not (0 <= sel < len(items)):
            return gr.update(value=items, selected_index=st.get("selected")), st
        items.pop(sel)
        if not items:
            st["items"] = []; st["selected"] = None
            return gr.update(value=[], selected_index=None), st
        new_sel = min(sel, len(items) - 1)
        st["items"] = items; st["selected"] = new_sel
        # return gr.update(value=items, selected_index=new_sel), st
        return gr.update(value=items), st

    def _on_move(self, delta: int, state: Dict[str, Any], gallery) :
        st = get_state(state); items: List[Any] = get_list(gallery); sel = st.get("selected", None)
        if sel is None or not (0 <= sel < len(items)):
            return gr.update(value=items, selected_index=sel), st
        j = sel + delta
        if j < 0 or j >= len(items):
            return gr.update(value=items, selected_index=sel), st
        items[sel], items[j] = items[j], items[sel]
        st["items"] = items; st["selected"] = j
        return gr.update(value=items, selected_index=j), st

    def _on_clear(self, state: Dict[str, Any]) :
        st = {"items": [], "selected": None, "single": get_state(state).get("single", False), "mode": self.media_mode}
        return gr.update(value=[], selected_index=0), st

    def _on_toggle_single(self, to_single: bool, state: Dict[str, Any]) :
        st = get_state(state); st["single"] = bool(to_single)
        items: List[Any] = list(st["items"]); sel = st.get("selected", None)
        if st["single"]:
            keep = items[sel] if (sel is not None and 0 <= sel < len(items)) else (items[-1] if items else None)
            items = [keep] if keep is not None else []
            sel = 0 if items else None
        st["items"] = items; st["selected"] = sel

        upload_update = gr.update(file_count=("single" if st["single"] else "multiple"))
        left_update   = gr.update(visible=not st["single"])
        right_update  = gr.update(visible=not st["single"])
        clear_update  = gr.update(visible=not st["single"])
        gallery_update= gr.update(value=items, selected_index=sel)

        return upload_update, left_update, right_update, clear_update, gallery_update, st

    # ---------------- build & wire ----------------

    def mount(self, parent: Optional[gr.Blocks | gr.Group | gr.Row | gr.Column] = None, update_form = False):
        if parent is not None:
            with parent:
                col = self._build_ui()
        else:
            col = self._build_ui()
        if not update_form:
            self._wire_events()
        return col

    def _build_ui(self) -> gr.Column:
        with gr.Column(elem_id=self.elem_id, elem_classes=self.elem_classes) as col:
            self.container = col

            self.state = gr.State(dict(self._initial_state))

            self.gallery = gr.Gallery(
                label=self.label,
                value=self._initial_state["items"],
                height=self.height,
                columns=self.columns,
                show_label=self.show_label,
                preview= True,
                # type="pil",
                file_types= list(IMAGE_EXTS) if self.media_mode == "image" else list(VIDEO_EXTS), 
                selected_index=self._initial_state["selected"],  # server-side selection
            )

            # One-line controls
            exts = sorted(IMAGE_EXTS if self.media_mode == "image" else VIDEO_EXTS) if self.accept_filter else None
            with gr.Row(equal_height=True, elem_classes=["amg-controls"]):
                self.upload_btn = gr.UploadButton(
                    "Set" if self._initial_state["single"] else "Add",
                    file_types=exts,
                    file_count=("single" if self._initial_state["single"] else "multiple"),
                    variant="primary",
                    size="sm",
                    min_width=1,
                )
                self.btn_remove = gr.Button("Remove", size="sm", min_width=1)
                self.btn_left   = gr.Button("◀ Left",  size="sm", visible=not self._initial_state["single"], min_width=1)
                self.btn_right  = gr.Button("Right ▶", size="sm", visible=not self._initial_state["single"], min_width=1)
                self.btn_clear  = gr.Button("Clear",   variant="secondary", size="sm", visible=not self._initial_state["single"], min_width=1)

        return col

    def _wire_events(self):
        # Selection: mirror into state and keep gallery.selected_index in sync
        self.gallery.select(
            self._on_select,
            inputs=[self.state, self.gallery],
            outputs=[self.state],
        )

        # Gallery value changed by user actions (click-to-add, drag-drop, internal remove, etc.)
        self.gallery.change(
            self._on_gallery_change,
            inputs=[self.gallery, self.state],
            outputs=[self.gallery, self.state],
        )

        # Add via UploadButton
        self.upload_btn.upload(
            self._on_add,
            inputs=[self.upload_btn, self.state, self.gallery],
            outputs=[self.gallery, self.state],
        )

        # Remove selected
        self.btn_remove.click(
            self._on_remove,
            inputs=[self.state, self.gallery],
            outputs=[self.gallery, self.state],
        )

        # Reorder using selected index, keep same item selected
        self.btn_left.click(
            lambda st, gallery: self._on_move(-1, st, gallery),
            inputs=[self.state, self.gallery],
            outputs=[self.gallery, self.state],
        )
        self.btn_right.click(
            lambda st, gallery: self._on_move(+1, st, gallery),
            inputs=[self.state, self.gallery],
            outputs=[self.gallery, self.state],
        )

        # Clear all
        self.btn_clear.click(
            self._on_clear,
            inputs=[self.state],
            outputs=[self.gallery, self.state],
        )

    # ---------------- public API ----------------

    def set_one_image_mode(self, enabled: bool = True):
        """Toggle single-image mode at runtime."""
        return (
            self._on_toggle_single,
            [gr.State(enabled), self.state],
            [self.upload_btn, self.btn_left, self.btn_right, self.btn_clear, self.gallery, self.state],
        )

    def get_toggable_elements(self):
        return [self.upload_btn, self.btn_left, self.btn_right, self.btn_clear, self.gallery, self.state]

# import gradio as gr

# with gr.Blocks() as demo:
#     amg = AdvancedMediaGallery(media_mode="image", height=190, columns=8)
#     amg.mount()
#     g = amg.gallery
#     # buttons to switch modes live (optional)
#     def process(g):
#         pass
#     with gr.Row():
#         gr.Button("toto").click(process, g)
#         gr.Button("ONE image").click(*amg.set_one_image_mode(True))
#         gr.Button("MULTI image").click(*amg.set_one_image_mode(False))

# demo.launch()
