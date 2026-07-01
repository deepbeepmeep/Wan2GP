from __future__ import annotations

import os
import re
import time
import copy
from pathlib import Path

ENABLE_GRADIO_MODEL_SWITCH_MONKEYPATCH = False
_OUTPUT_PROP_HISTORY_ATTR = "_wangp_output_prop_history"

_ORIGINAL_OUTPUT_UPDATER = 'async function pl(S,J){const K=u.find(de=>de.id===J);if(!K)return;const pe=K.outputs,R=S?.map((de,he)=>({id:pe[he],prop:"value_is_output",value:!0}));T(R),await Rn();const oe=[];S?.forEach((de,he)=>{if(typeof de=="object"&&de!==null&&de.__type__==="update")for(const[bt,Qt]of Object.entries(de))bt!=="__type__"&&oe.push({id:pe[he],prop:bt,value:Qt});else oe.push({id:pe[he],prop:"value",value:de})}),T(oe),await Rn()}'
_PATCHED_OUTPUT_UPDATER = 'async function pl(S,J){const K=u.find(de=>de.id===J);if(!K)return;const pe=K.outputs,R=[],oe=[];S?.forEach((de,he)=>{if(typeof de=="object"&&de!==null&&de.__type__==="update"){if("value"in de)R.push({id:pe[he],prop:"value_is_output",value:!0});for(const[bt,Qt]of Object.entries(de))bt!=="__type__"&&oe.push({id:pe[he],prop:bt,value:Qt})}else R.push({id:pe[he],prop:"value_is_output",value:!0}),oe.push({id:pe[he],prop:"value",value:de})});if(R.length){T(R);await Rn()}if(oe.length){T(oe);await Rn()}}'
_DEBUG_OUTPUT_UPDATER = 'async function pl(S,J){const nt=performance.now(),K=u.find(de=>de.id===J);if(!K)return;const pe=K.outputs,R=[],oe=[];S?.forEach((de,he)=>{if(typeof de=="object"&&de!==null&&de.__type__==="update"){if("value"in de)R.push({id:pe[he],prop:"value_is_output",value:!0});for(const[bt,Qt]of Object.entries(de))bt!=="__type__"&&oe.push({id:pe[he],prop:bt,value:Qt})}else R.push({id:pe[he],prop:"value_is_output",value:!0}),oe.push({id:pe[he],prop:"value",value:de})});if(R.length){T(R);await Rn()}const rt=performance.now();if(oe.length){T(oe);await Rn()}const et=Math.round((performance.now()-nt)*10)/10,ot=Math.round((rt-nt)*10)/10;if((S?.length||0)>50||et>20)console.info("[WanGP ui-perf] gradio.output_update fn="+J+" outputs="+(S?.length||0)+" value_marks="+R.length+" prop_updates="+oe.length+" mark_ms="+ot+" total_ms="+et)}'
_ORIGINAL_WAIT_THEN_TRIGGER = 'function Jt(S,J=null,K=null){let pe=()=>{};function R(){pe()}i?pe=Se.subscribe(oe=>{oe||Rn().then(()=>{is(S,J,K),R()})}):is(S,J,K)}'
_PATCHED_WAIT_THEN_TRIGGER = 'function Jt(S,J=null,K=null){let pe=()=>{};function R(){pe()}const oe=u.find(de=>de.id===S),de=()=>{is(S,J,K)};if(oe?.wangp_fast_chain&&oe.queue===!1){Promise.resolve().then(()=>de());return}i?pe=Se.subscribe(he=>{he||Rn().then(()=>{de(),R()})}):de()}'
_ORIGINAL_EVENT_TRIGGER = 's[pe]?.[R]?.forEach(he=>{requestAnimationFrame(()=>{Jt(he,pe,oe)})})'
_PATCHED_EVENT_TRIGGER = 's[pe]?.[R]?.forEach(he=>{const Qt=u.find(Mt=>Mt.id===he);Qt?.wangp_fast_chain&&Qt.queue===!1?Jt(he,pe,oe):requestAnimationFrame(()=>{Jt(he,pe,oe)})})'
_ORIGINAL_DATA_STATUS_UPDATE = 'R.pending_request&&R.final_event&&(R.pending_request=!1,Qt(R.final_event,R.connection=="stream")),R.pending_request=!1,pl(Ee,ne),Mn(n)'
_PATCHED_DATA_STATUS_UPDATE = 'R.pending_request&&R.final_event&&(R.pending_request=!1,Qt(R.final_event,R.connection=="stream")),R.pending_request=!1,pl(Ee,ne),R.wangp_skip_loading_status||Mn(n)'
_ORIGINAL_STATUS_UPDATE = 'ie.update({...ne,time_limit:ne.time_limit,status:ne.stage,progress:ne.progress_data,fn_index:Ee}),Mn(n),'
_PATCHED_STATUS_UPDATE = '(R.wangp_skip_loading_status?0:(ie.update({...ne,time_limit:ne.time_limit,status:ne.stage,progress:ne.progress_data,fn_index:Ee}),Mn(n))),'
_ORIGINAL_PROGRESS_TIMER = 'function ue(){Oe(()=>{e(26,B=(performance.now()-U)/1e3),K&&ue()})}'
_PATCHED_PROGRESS_TIMER = 'function ue(){Oe(()=>{const d=J&&J.offsetParent!==null&&J.getClientRects().length>0;if(!d){K=!1;return}e(26,B=(performance.now()-U)/1e3),K&&ue()})}'
_PATCH_INIT = r''';(()=>{window.__WANGP_MODEL_SWITCH_PATCH_INIT=1})();'''
_PATCH_ACTIVE_LOG = ';console.info("[WanGP] Gradio output update patch active");'

_installed = False
_verbose = False
_warned_assets: set[str] = set()
_progress_cap_logged: set[int] = set()
_direct_queue_logged: set[int] = set()
_ASSET_CACHE_BUST_MTIME = Path(__file__).stat().st_mtime
_ASSET_CACHE_BUST_TOKEN = f"wangp-{int(_ASSET_CACHE_BUST_MTIME)}"
_ASSET_CACHE_BUST_PARAM = f"__wangp_ui={_ASSET_CACHE_BUST_TOKEN}"
_HTML_BOOT_JS_RE = re.compile(r'(?P<prefix>\bsrc=["\'])(?P<url>(?:\.?/)?assets/index-[^"\']+\.js)(?P<suffix>["\'])')

_MODEL_SWITCH_DIRECT_FN_NAMES = {
    "record_image_mode_tab",
    "save_inputs",
    "switch_image_mode",
    "validate_wizard_prompt",
    "goto_model_type",
    "goto_model_type_with_filter",
    "change_model_from_target",
    "fill_inputs",
    "_refresh_from_main",
    "_media_visibility_updates",
    "refresh_multiplier",
    "refresh_video_prompt_type_video_mask",
    "refresh_video_prompt_type_video_guide",
}


def _same_value(old_value, new_value) -> bool:
    if old_value is new_value and not isinstance(old_value, (dict, list, set, tuple)):
        return True
    try:
        result = old_value == new_value
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    if result.__class__.__name__ == "bool_":
        return bool(result)
    return False


def _snapshot_value(value):
    if isinstance(value, (dict, list, set, tuple)):
        try:
            return copy.deepcopy(value)
        except Exception:
            return value
    return value


def _get_output_prop_history(state):
    history = getattr(state, _OUTPUT_PROP_HISTORY_ATTR, None)
    if history is None:
        history = {}
        setattr(state, _OUTPUT_PROP_HISTORY_ATTR, history)
    return history


def _record_output_props(block_fn, output, props_by_id):
    for index, value in enumerate(output):
        if not (isinstance(value, dict) and value.get("__type__") == "update"):
            continue
        block = block_fn.outputs[index]
        block_props = props_by_id.setdefault(block._id, {})
        for key, prop_value in value.items():
            if key != "__type__":
                block_props[key] = _snapshot_value(prop_value)


def _prune_unchanged_output_props(block_fn, output, props_by_id):
    props_before = 0
    props_after = 0
    pruned = 0
    for index, value in enumerate(output):
        if not (isinstance(value, dict) and value.get("__type__") == "update"):
            continue
        block = block_fn.outputs[index]
        previous_props = props_by_id.get(block._id)
        if previous_props is None:
            continue
        for key in list(value):
            if key == "__type__":
                continue
            props_before += 1
            if key in previous_props and _same_value(previous_props[key], value[key]):
                del value[key]
                pruned += 1
        props_after += len(value) - (1 if value.get("__type__") == "update" else 0)
    return props_before, props_after, pruned


def _install_postprocess_prune_patch() -> bool:
    import gradio.blocks as gradio_blocks

    original_postprocess_data = gradio_blocks.Blocks.postprocess_data
    if getattr(original_postprocess_data, "_wangp_model_switch_patch_installed", False):
        return True

    async def patched_postprocess_data(self, block_fn, predictions, state):
        session_state = state or gradio_blocks.SessionState(self)
        props_by_id = _get_output_prop_history(session_state)
        start_time = time.perf_counter()
        output = await original_postprocess_data(self, block_fn, predictions, session_state)
        if isinstance(output, list) and props_by_id:
            props_before, props_after, pruned = _prune_unchanged_output_props(block_fn, output, props_by_id)
            _record_output_props(block_fn, output, props_by_id)
            if _verbose and (pruned > 0 or props_before > 200):
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
                print(f"[WanGP ui-perf] gradio.postprocess_prune fn={getattr(block_fn, 'name', '')} outputs={len(output)} props_before={props_before} props_after={props_after} pruned={pruned} elapsed_ms={elapsed_ms}", flush=True)
        elif isinstance(output, list):
            _record_output_props(block_fn, output, props_by_id)
        return output

    patched_postprocess_data._wangp_model_switch_patch_installed = True
    patched_postprocess_data._wangp_original_postprocess_data = original_postprocess_data
    gradio_blocks.Blocks.postprocess_data = patched_postprocess_data
    return True


def _install_large_progress_patch() -> bool:
    import gradio.blocks as gradio_blocks

    original_get_config = gradio_blocks.BlockFunction.get_config
    if getattr(original_get_config, "_wangp_model_switch_patch_installed", False):
        return True

    def patched_get_config(self):
        config = original_get_config(self)
        name = self.name or ""
        if (
            config.get("backend_fn")
            and config.get("queue") is not False
            and name in _MODEL_SWITCH_DIRECT_FN_NAMES
            and not self.types_generator
            and self.connection == "sse"
        ):
            config["queue"] = False
            config["wangp_fast_chain"] = True
            config["wangp_skip_loading_status"] = True
            if _verbose and self._id not in _direct_queue_logged:
                _direct_queue_logged.add(self._id)
                print(f"[WanGP ui-perf] gradio.direct_queue fn={self.name} id={self._id} outputs={len(config.get('outputs') or [])}", flush=True)
        elif (
            config.get("backend_fn")
            and config.get("queue") is not False
            and name == "<lambda>"
            and self.show_progress == "hidden"
            and len(config.get("inputs") or []) == 1
            and len(config.get("outputs") or []) <= 1
            and not self.types_generator
            and self.connection == "sse"
        ):
            config["queue"] = False
            config["wangp_fast_chain"] = True
            config["wangp_skip_loading_status"] = True
            if _verbose and self._id not in _direct_queue_logged:
                _direct_queue_logged.add(self._id)
                print(f"[WanGP ui-perf] gradio.direct_queue fn={name} id={self._id} outputs={len(config.get('outputs') or [])}", flush=True)
        outputs = config.get("outputs") or []
        if config.get("show_progress") == "full" and config.get("show_progress_on") is None and len(outputs) > 64:
            config["show_progress_on"] = outputs[:1]
            if _verbose and self._id not in _progress_cap_logged:
                _progress_cap_logged.add(self._id)
                print(f"[WanGP ui-perf] gradio.progress_cap fn={self.name} id={self._id} outputs={len(outputs)} show_progress_on={config['show_progress_on']}", flush=True)
        return config

    patched_get_config._wangp_model_switch_patch_installed = True
    patched_get_config._wangp_original_get_config = original_get_config
    gradio_blocks.BlockFunction.get_config = patched_get_config
    return True


def _patched_asset_text(path_text: str) -> str | None:
    source = Path(path_text).read_text(encoding="utf-8")
    patched = False
    if _ORIGINAL_OUTPUT_UPDATER not in source:
        if path_text not in _warned_assets:
            _warned_assets.add(path_text)
            print(f"[WanGP] Gradio output update patch skipped; updater signature not found in {path_text}", flush=True)
    else:
        updater = _DEBUG_OUTPUT_UPDATER if _verbose else _PATCHED_OUTPUT_UPDATER
        active_log = _PATCH_ACTIVE_LOG if _verbose else ""
        source = source.replace(_ORIGINAL_OUTPUT_UPDATER, updater + _PATCH_INIT + active_log, 1)
        patched = True
    if _ORIGINAL_WAIT_THEN_TRIGGER in source:
        source = source.replace(_ORIGINAL_WAIT_THEN_TRIGGER, _PATCHED_WAIT_THEN_TRIGGER, 1)
        patched = True
    elif path_text not in _warned_assets:
        _warned_assets.add(path_text)
        print(f"[WanGP] Gradio fast-chain patch skipped; trigger signature not found in {path_text}", flush=True)
    if _ORIGINAL_EVENT_TRIGGER in source:
        source = source.replace(_ORIGINAL_EVENT_TRIGGER, _PATCHED_EVENT_TRIGGER, 1)
        patched = True
    elif path_text not in _warned_assets:
        _warned_assets.add(path_text)
        print(f"[WanGP] Gradio event-trigger patch skipped; trigger signature not found in {path_text}", flush=True)
    if _ORIGINAL_DATA_STATUS_UPDATE in source:
        source = source.replace(_ORIGINAL_DATA_STATUS_UPDATE, _PATCHED_DATA_STATUS_UPDATE, 1)
        patched = True
    elif path_text not in _warned_assets:
        _warned_assets.add(path_text)
        print(f"[WanGP] Gradio data-status patch skipped; status signature not found in {path_text}", flush=True)
    if _ORIGINAL_STATUS_UPDATE in source:
        source = source.replace(_ORIGINAL_STATUS_UPDATE, _PATCHED_STATUS_UPDATE, 1)
        patched = True
    elif path_text not in _warned_assets:
        _warned_assets.add(path_text)
        print(f"[WanGP] Gradio status patch skipped; status signature not found in {path_text}", flush=True)
    return source if patched else None


def _patched_progress_timer_asset_text(path_text: str) -> str | None:
    source = Path(path_text).read_text(encoding="utf-8")
    return _patch_progress_timer_source(source)


def _patch_progress_timer_source(source: str) -> str | None:
    if _ORIGINAL_PROGRESS_TIMER in source:
        return source.replace(_ORIGINAL_PROGRESS_TIMER, _PATCHED_PROGRESS_TIMER, 1)
    return None


def _cache_bust_url(url: str) -> str:
    if "__wangp_ui=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{_ASSET_CACHE_BUST_PARAM}"


def _cache_bust_html_assets(source: str) -> str:
    return _HTML_BOOT_JS_RE.sub(lambda match: f"{match.group('prefix')}{_cache_bust_url(match.group('url'))}{match.group('suffix')}", source)


def _install_template_cache_bust_patch(gradio_routes) -> bool:
    original_template_response = gradio_routes.templates.TemplateResponse
    if getattr(original_template_response, "_wangp_model_switch_patch_installed", False):
        return True

    def patched_template_response(*args, **kwargs):
        response = original_template_response(*args, **kwargs)
        body = getattr(response, "body", None)
        if body:
            try:
                source = body.decode("utf-8")
            except UnicodeDecodeError:
                return response
            patched = _cache_bust_html_assets(source)
            if patched != source:
                response.body = patched.encode("utf-8")
                response.headers["content-length"] = str(len(response.body))
                response.headers["cache-control"] = "no-store"
                response.headers["x-wangp-gradio-patch"] = "html-asset-cache-bust"
        return response

    patched_template_response._wangp_model_switch_patch_installed = True
    patched_template_response._wangp_original_template_response = original_template_response
    gradio_routes.templates.TemplateResponse = patched_template_response
    return True



def install(verbose: bool = False) -> bool:
    global _installed, _verbose
    _verbose = verbose
    if os.getenv("WANGP_DISABLE_GRADIO_MODEL_SWITCH_PATCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        if verbose:
            print("[WanGP] Gradio model switch patches disabled by WANGP_DISABLE_GRADIO_MODEL_SWITCH_PATCH", flush=True)
        return False
    if _installed:
        return True
    if not ENABLE_GRADIO_MODEL_SWITCH_MONKEYPATCH:
        return False

    import gradio.routes as gradio_routes
    from fastapi.responses import Response

    original_file_response = gradio_routes.FileResponse
    if getattr(original_file_response, "_wangp_model_switch_patch_installed", False):
        _installed = True
        return True

    def patched_file_response(path, *args, **kwargs):
        path_text = str(path)
        filename = Path(path_text).name
        patch_names = []
        patched_text = None
        if filename.startswith("Blocks-") and filename.endswith(".js"):
            patched_text = _patched_asset_text(path_text)
            if patched_text is not None:
                patch_names.append("output-update")
        # Keep dynamic imports on their original Vite URLs. The JS responses are
        # already served with no-store below; rewriting dynamic imports creates a
        # second module identity for chunks that also appear in Vite's preload map.
        if filename.endswith(".js"):
            progress_timer_text = _patch_progress_timer_source(patched_text) if patched_text is not None else _patched_progress_timer_asset_text(path_text)
            if progress_timer_text is not None:
                patched_text = progress_timer_text
                patch_names.append("progress-timer")
        if patched_text is not None:
            headers = dict(kwargs.pop("headers", None) or {})
            headers["Cache-Control"] = "no-store"
            headers["X-WanGP-Gradio-Patch"] = ",".join(patch_names)
            return Response(patched_text, media_type="application/javascript", headers=headers)
        if filename.endswith(".js"):
            headers = dict(kwargs.pop("headers", None) or {})
            headers["Cache-Control"] = "no-store"
            kwargs["headers"] = headers
        return original_file_response(path, *args, **kwargs)

    patched_file_response._wangp_model_switch_patch_installed = True
    patched_file_response._wangp_original_file_response = original_file_response
    gradio_routes.FileResponse = patched_file_response
    _install_template_cache_bust_patch(gradio_routes)
    _install_large_progress_patch()
    _install_postprocess_prune_patch()
    _installed = True
    if verbose:
        print("[WanGP] Gradio output update patches installed", flush=True)
    return True


__all__ = [
    "ENABLE_GRADIO_MODEL_SWITCH_MONKEYPATCH",
    "install",
]
