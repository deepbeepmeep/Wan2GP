import gradio as gr
import os
import json
import html
from collections import defaultdict
from shared.utils.plugins import WAN2GPPlugin
from shared.utils import files_locator as fl
from mmgp import quant_router


class modelsManagerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self._node_map = {}
        self._tree_data = []
        self._model_files = {}
        self._model_primary = {}
        self._model_variants = {}
        self._variant_to_model = {}
        self._model_base_label = {}
        self._model_family_label = {}
        self._model_usage_ids = {}
        self._usage_files = {}
        self._file_usage = {}
        self._file_usage_models = {}
        self._file_info = {}
        self._basename_index = {}
        self._stats_total = 0
        self._stats_by_drive = []
        self._errors = []
        self._repo_root = os.path.abspath(os.getcwd())

    def setup_ui(self):
        self.request_global("get_model_def")
        self.request_global("get_model_name")
        self.request_global("get_model_family")
        self.request_global("get_parent_model_type")
        self.request_global("get_base_model_type")
        self.request_global("get_model_handler")
        self.request_global("get_model_recursive_prop")
        self.request_global("get_model_filename")
        self.request_global("get_local_model_filename")
        self.request_global("get_lora_dir")
        self.request_global("compact_name")
        self.request_global("create_models_hierarchy")
        self.request_global("families_infos")
        self.request_global("models_def")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("text_encoder_quantization")
        self.request_global("displayed_model_types")
        self.request_global("transformer_types")

        self.add_custom_js(self._get_js())

        self.add_tab(
            tab_id="model_manager",
            label="Models Manager",
            component_constructor=self.create_model_ui,
        )

    def create_model_ui(self):
        with gr.Column():
            gr.Markdown("### Models Manager")
            gr.Markdown(
                "Sizes are unique GB recoverable on delete, plus shared GB used outside the branch."
            )
            with gr.Row():
                with gr.Column(scale=3, min_width=550):
                    self.tree_html = gr.HTML()
                with gr.Column(scale=2, min_width=350, elem_classes="ckpt-right-column"):
                    self.view_html = gr.HTML()

            self.refresh_button = gr.Button(
                "Refresh", variant="secondary", elem_id="ckpt_refresh_btn"
            )

            with gr.Column(visible=False):
                self.action_input = gr.Textbox(elem_id="model_action_input")

        self.refresh_button.click(
            fn=self._show_loading,
            inputs=[],
            outputs=[self.tree_html, self.view_html, self.refresh_button],
            show_progress="hidden",
        ).then(
            fn=self._build_tree,
            inputs=[],
            outputs=[self.tree_html, self.view_html, self.refresh_button],
            show_progress="full",
        )

        self.action_input.change(
            fn=self._handle_action,
            inputs=[self.action_input],
            outputs=[self.tree_html, self.view_html, self.refresh_button],
            show_progress="hidden",
        )

        self.on_tab_outputs = [self.tree_html, self.view_html, self.refresh_button]
        return self.tree_html

    def on_tab_select(self, state: dict):
        return self._show_loading()

    def _get_js(self):
        return """
        function modelRoot() {
            if (window.gradioApp) {
                return window.gradioApp();
            }
            const app = document.querySelector('gradio-app');
            return app ? (app.shadowRoot || app) : document;
        }

        window.updatemodelInput = function(elem_id, value) {
            const root = modelRoot();
            const input = root.querySelector(`#${elem_id} textarea, #${elem_id} input`);
            if (input) {
                input.value = value;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                return true;
            }
            return false;
        };

        function ensuremodelModal() {
            let backdrop = document.getElementById('ckpt-delete-modal');
            if (backdrop) return backdrop;
            backdrop = document.createElement('div');
            backdrop.id = 'ckpt-delete-modal';
            backdrop.className = 'ckpt-modal-backdrop';
            backdrop.innerHTML = `
                <div class="ckpt-modal">
                    <div class="ckpt-modal-title">Confirm Delete</div>
                    <div class="ckpt-modal-body" id="ckpt-delete-modal-body"></div>
                    <div class="ckpt-modal-actions">
                        <button class="ckpt-modal-btn" id="ckpt-delete-cancel">Cancel</button>
                        <button class="ckpt-modal-btn danger" id="ckpt-delete-unique">Delete Unique</button>
                        <button class="ckpt-modal-btn danger" id="ckpt-delete-all">Delete Unique + Shared</button>
                    </div>
                </div>
            `;
            document.body.appendChild(backdrop);
            const cancelBtn = backdrop.querySelector('#ckpt-delete-cancel');
            const uniqueBtn = backdrop.querySelector('#ckpt-delete-unique');
            const allBtn = backdrop.querySelector('#ckpt-delete-all');
            cancelBtn.addEventListener('click', () => {
                backdrop.style.display = 'none';
                backdrop.dataset.payloadUnique = '';
                backdrop.dataset.payloadAll = '';
            });
            uniqueBtn.addEventListener('click', () => {
                const payload = backdrop.dataset.payloadUnique || '';
                if (payload) {
                    window.updatemodelInput("model_action_input", payload);
                }
                backdrop.style.display = 'none';
                backdrop.dataset.payloadUnique = '';
                backdrop.dataset.payloadAll = '';
            });
            allBtn.addEventListener('click', () => {
                const payload = backdrop.dataset.payloadAll || '';
                if (payload) {
                    window.updatemodelInput("model_action_input", payload);
                }
                backdrop.style.display = 'none';
                backdrop.dataset.payloadUnique = '';
                backdrop.dataset.payloadAll = '';
            });
            backdrop.addEventListener('click', (evt) => {
                if (evt.target === backdrop) {
                    backdrop.style.display = 'none';
                    backdrop.dataset.payloadUnique = '';
                    backdrop.dataset.payloadAll = '';
                }
            });
            return backdrop;
        }

        function showmodelModal(message, payloadUnique, payloadAll, uniqueSize, totalSize) {
            const backdrop = ensuremodelModal();
            const body = backdrop.querySelector('#ckpt-delete-modal-body');
            const uniqueBtn = backdrop.querySelector('#ckpt-delete-unique');
            const allBtn = backdrop.querySelector('#ckpt-delete-all');
            body.textContent = message;
            backdrop.dataset.payloadUnique = payloadUnique || '';
            backdrop.dataset.payloadAll = payloadAll || '';
            uniqueBtn.textContent = uniqueSize ? `Delete Unique (${uniqueSize})` : 'Delete Unique';
            allBtn.textContent = totalSize ? `Delete Unique + Shared (${totalSize})` : 'Delete Unique + Shared';
            backdrop.style.display = 'flex';
        }

        window.handlemodelAction = function(button, action) {
            if (!button) return;
            const nodeId = button.dataset.nodeId;
            const nodeLabel = button.dataset.nodeLabel || nodeId;
            const uniqueSize = button.dataset.uniqueSize || "0 B";
            const totalSize = button.dataset.totalSize || uniqueSize;
            if (!nodeId) return;
            if (action === "delete") {
                const payloadUnique = JSON.stringify({
                    action: action,
                    node_id: nodeId,
                    delete_shared: false,
                    ts: Date.now()
                });
                const payloadAll = JSON.stringify({
                    action: action,
                    node_id: nodeId,
                    delete_shared: true,
                    ts: Date.now()
                });
                showmodelModal(
                    `Choose delete scope for "${nodeLabel}": unique only ${uniqueSize}, or unique + shared ${totalSize}.`,
                    payloadUnique,
                    payloadAll,
                    uniqueSize,
                    totalSize
                );
                return;
            }
            const payload = JSON.stringify({
                action: action,
                node_id: nodeId,
                ts: Date.now()
            });
            window.updatemodelInput("model_action_input", payload);
        };

        window.handlemodelRowClick = function(evt, row) {
            if (evt) evt.stopPropagation();
            if (!row) return;
            const nodeId = row.dataset.nodeId;
            if (!nodeId) return;
            const payload = JSON.stringify({
                action: "view",
                node_id: nodeId,
                ts: Date.now()
            });
            window.updatemodelInput("model_action_input", payload);
        };

        function ismodelTabActive() {
            const root = modelRoot();
            const tabs = root.querySelectorAll('button[role="tab"]');
            for (const tab of tabs) {
                if (tab.getAttribute('aria-selected') === 'true') {
                    const label = (tab.textContent || '').trim();
                    if (label === 'Models Manager') return true;
                }
            }
            return false;
        }

        function triggermodelRefresh() {
            const root = modelRoot();
            const btn = root.querySelector('#ckpt_refresh_btn button') || root.querySelector('#ckpt_refresh_btn');
            if (btn) btn.click();
        }

        function applymodelSticky() {
            const root = modelRoot();
            const sticky = root.querySelector('.ckpt-view-sticky');
            if (!sticky) return;
            sticky.style.position = 'sticky';
            sticky.style.top = '16px';
            sticky.style.alignSelf = 'flex-start';
            let parent = sticky.parentElement;
            let steps = 0;
            while (parent && steps < 6) {
                const style = window.getComputedStyle(parent);
                if (style.overflow && style.overflow !== 'visible') {
                    parent.style.overflow = 'visible';
                }
                parent = parent.parentElement;
                steps += 1;
            }
            if (root && root.host && root.host.style) {
                const hostStyle = window.getComputedStyle(root.host);
                if (hostStyle.overflow && hostStyle.overflow !== 'visible') {
                    root.host.style.overflow = 'visible';
                }
            }
        }

        let ckptFloating = {
            sticky: null,
            scrollEl: null,
            handler: null,
            resizeHandler: null,
            anchorTop: 0,
            topPadding: 16,
        };

        function findmodelScrollContainer(element) {
            let current = element;
            while (current) {
                const style = window.getComputedStyle(current);
                const overflowY = style.overflowY;
                if ((overflowY === 'auto' || overflowY === 'scroll') && current.scrollHeight > current.clientHeight + 1) {
                    return current;
                }
                current = current.parentElement;
            }
            const root = modelRoot();
            const host = root && root.host ? root.host : null;
            current = host;
            while (current) {
                const style = window.getComputedStyle(current);
                const overflowY = style.overflowY;
                if ((overflowY === 'auto' || overflowY === 'scroll') && current.scrollHeight > current.clientHeight + 1) {
                    return current;
                }
                current = current.parentElement;
            }
            return window;
        }

        function computemodelAnchorTop(sticky, scrollEl) {
            if (scrollEl === window) {
                return sticky.getBoundingClientRect().top + window.scrollY;
            }
            const rect = sticky.getBoundingClientRect();
            const containerRect = scrollEl.getBoundingClientRect();
            return rect.top - containerRect.top + scrollEl.scrollTop;
        }

        function clearmodelFloating(sticky) {
            if (!sticky) return;
            sticky.style.transform = '';
            sticky.style.willChange = '';
        }

        function updatemodelFloating() {
            const sticky = ckptFloating.sticky;
            if (!sticky) return;
            const scrollEl = ckptFloating.scrollEl || window;
            const scrollTop = scrollEl === window ? window.scrollY : scrollEl.scrollTop;
            let maxTranslate = Infinity;
            if (scrollEl === window) {
                const doc = document.documentElement;
                maxTranslate = Math.max(
                    0,
                    (doc ? doc.scrollHeight : 0) - sticky.offsetHeight - ckptFloating.topPadding - ckptFloating.anchorTop
                );
            } else {
                maxTranslate = Math.max(
                    0,
                    scrollEl.scrollHeight - sticky.offsetHeight - ckptFloating.topPadding - ckptFloating.anchorTop
                );
            }
            const offset = scrollTop + ckptFloating.topPadding - ckptFloating.anchorTop;
            const translate = Math.max(0, Math.min(offset, maxTranslate));
            if (translate > 0) {
                sticky.style.transform = `translateY(${translate}px)`;
                sticky.style.willChange = 'transform';
            } else {
                clearmodelFloating(sticky);
            }
        }

        function teardownmodelFloating() {
            if (ckptFloating.scrollEl && ckptFloating.handler) {
                if (ckptFloating.scrollEl === window) {
                    window.removeEventListener('scroll', ckptFloating.handler);
                } else {
                    ckptFloating.scrollEl.removeEventListener('scroll', ckptFloating.handler);
                }
            }
            if (ckptFloating.resizeHandler) {
                window.removeEventListener('resize', ckptFloating.resizeHandler);
            }
            clearmodelFloating(ckptFloating.sticky);
            if (ckptFloating.sticky) {
                ckptFloating.sticky.style.position = '';
                ckptFloating.sticky.style.top = '';
            }
            ckptFloating.sticky = null;
            ckptFloating.scrollEl = null;
            ckptFloating.handler = null;
            ckptFloating.resizeHandler = null;
            ckptFloating.anchorTop = 0;
        }

        function setupmodelFloating() {
            const root = modelRoot();
            const sticky = root.querySelector('.ckpt-view-sticky');
            if (!sticky) return;
            if (ckptFloating.sticky === sticky) {
                updatemodelFloating();
                return;
            }
            teardownmodelFloating();
            ckptFloating.sticky = sticky;
            sticky.style.position = 'relative';
            sticky.style.top = '0px';
            ckptFloating.scrollEl = findmodelScrollContainer(sticky);
            ckptFloating.anchorTop = computemodelAnchorTop(sticky, ckptFloating.scrollEl);
            ckptFloating.handler = () => updatemodelFloating();
            if (ckptFloating.scrollEl === window) {
                window.addEventListener('scroll', ckptFloating.handler, { passive: true });
            } else {
                ckptFloating.scrollEl.addEventListener('scroll', ckptFloating.handler, { passive: true });
            }
            ckptFloating.resizeHandler = () => {
                ckptFloating.anchorTop = computemodelAnchorTop(sticky, ckptFloating.scrollEl);
                updatemodelFloating();
            };
            window.addEventListener('resize', ckptFloating.resizeHandler);
            updatemodelFloating();
        }

        function observemodelView() {
            const root = modelRoot();
            const column = root.querySelector('.ckpt-right-column');
            if (!column || column.dataset.ckptObserved === '1') return;
            column.dataset.ckptObserved = '1';
            const observer = new MutationObserver(() => {
                applymodelSticky();
                setupmodelFloating();
            });
            observer.observe(column, { childList: true, subtree: true });
        }

        let ckptTabWasActive = false;
        let ckptRefreshBusy = false;
        let ckptRefreshStartedAt = 0;

        function waitForRefreshReady() {
            const root = modelRoot();
            const treeRoot = root.querySelector('#ckpt_tree_root');
            if (treeRoot && treeRoot.dataset.status === 'ready') {
                ckptRefreshBusy = false;
                applymodelSticky();
                setupmodelFloating();
                return;
            }
            if (ckptRefreshStartedAt && Date.now() - ckptRefreshStartedAt > 60000) {
                ckptRefreshBusy = false;
                applymodelSticky();
                setupmodelFloating();
                return;
            }
            setTimeout(waitForRefreshReady, 400);
        }

        function startmodelBuild() {
            if (ckptRefreshBusy) return;
            ckptRefreshBusy = true;
            ckptRefreshStartedAt = Date.now();
            triggermodelRefresh();
            setTimeout(waitForRefreshReady, 200);
        }

        function handleTabChange() {
            const active = ismodelTabActive();
            if (active && !ckptTabWasActive) {
                startmodelBuild();
            }
            if (active) {
                observemodelView();
                setTimeout(applymodelSticky, 200);
                setTimeout(setupmodelFloating, 260);
            }
            ckptTabWasActive = active;
        }

        function initmodelBridge() {
            const root = modelRoot();
            const tabs = root.querySelectorAll('button[role="tab"]');
            if (!tabs || tabs.length === 0) {
                setTimeout(initmodelBridge, 400);
                return;
            }
            const tabObserver = new MutationObserver(handleTabChange);
            tabObserver.observe(root, { attributes: true, subtree: true, attributeFilter: ['aria-selected', 'class'] });
            handleTabChange();
        }

        initmodelBridge();
        """

    def _show_loading(self):
        tree_loading = self._build_loading_html()
        view_hidden = gr.update(value="", visible=False)
        refresh_hidden = gr.update(visible=False)
        return tree_loading, view_hidden, refresh_hidden

    def _build_tree(self):
        self._build_cache()
        tree_html = self._build_tree_html()
        view_html = self._build_empty_view_html()
        view_visible = gr.update(value=view_html, visible=True)
        refresh_visible = gr.update(visible=True)
        return tree_html, view_visible, refresh_visible

    def _handle_action(self, payload: str):
        if not payload:
            return gr.update(), gr.update(), gr.update()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return gr.update(), gr.update(), gr.update()

        action = data.get("action")
        node_id = data.get("node_id")
        if not action or not node_id:
            return gr.update(), gr.update(), gr.update()

        if not self._node_map:
            self._build_cache()

        if action == "view":
            view_html = self._build_view_html(node_id)
            return gr.update(), gr.update(value=view_html, visible=True), gr.update()

        if action == "delete":
            delete_shared = bool(data.get("delete_shared"))
            removed, missing, errors = self._delete_files_for_node(
                node_id, delete_shared=delete_shared
            )
            if errors:
                gr.Warning(f"Delete completed with errors: {len(errors)}")
            else:
                gr.Info(f"Deleted {len(removed)} files (missing: {len(missing)})")
            return self._build_tree()

        return gr.update(), gr.update(), gr.update()

    def _delete_files_for_node(self, node_id, delete_shared=False):
        node = self._node_map.get(node_id)
        if node is None:
            return [], [], []
        removed = []
        missing = []
        errors = []
        if delete_shared:
            target_files = node.get("files", set())
        else:
            target_files = node.get("unique_files", set())
        for path in sorted(target_files):
            if not os.path.isfile(path):
                missing.append(path)
                continue
            try:
                os.remove(path)
                removed.append(path)
            except OSError as exc:
                errors.append((path, str(exc)))
        return removed, missing, errors

    def _build_cache(self):
        self._node_map = {}
        self._tree_data = []
        self._model_files = {}
        self._model_primary = {}
        self._model_variants = {}
        self._variant_to_model = {}
        self._model_base_label = {}
        self._model_family_label = {}
        self._model_usage_ids = {}
        self._usage_files = {}
        self._file_usage = defaultdict(set)
        self._file_usage_models = defaultdict(set)
        self._file_info = {}
        self._basename_index = {}
        self._errors = []
        self._stats_total = 0
        self._stats_by_drive = []

        model_types = self._get_model_types()
        for model_type in model_types:
            try:
                variant_info = self._collect_model_variants(model_type)
            except Exception as exc:
                self._errors.append(f"{model_type}: {exc}")
                variant_info = {"variants": [], "primary_paths": [], "files": set()}
            variants = variant_info.get("variants", [])
            model_files = variant_info.get("files", set())
            self._model_variants[model_type] = variants
            self._model_primary[model_type] = variant_info.get("primary_paths", [])
            self._model_files[model_type] = model_files
            for path in model_files:
                self._file_usage_models[path].add(model_type)

        self._rebalance_other_shared_variants(model_types)

        self._usage_files = {}
        self._file_usage = defaultdict(set)
        self._variant_to_model = {}
        self._model_usage_ids = {}
        for model_type in model_types:
            usage_ids = set()
            for variant in self._model_variants.get(model_type, []):
                variant_id = variant["id"]
                usage_ids.add(variant_id)
                self._variant_to_model[variant_id] = model_type
                files = variant.get("files", set())
                self._usage_files[variant_id] = files
                for path in files:
                    self._file_usage[path].add(variant_id)
            self._model_usage_ids[model_type] = usage_ids

        for path in self._file_usage_models.keys():
            try:
                exists = os.path.isfile(path)
                size = os.path.getsize(path) if exists else 0
            except OSError:
                exists = False
                size = 0
            self._file_info[path] = {"size": size, "exists": exists}

        self._compute_global_stats()
        self._build_tree_structure(model_types)

    def _compute_global_stats(self):
        total = 0
        by_drive = defaultdict(int)
        for path, info in self._file_info.items():
            size = info.get("size", 0)
            if size <= 0:
                continue
            total += size
            drive = self._get_drive_label(path)
            by_drive[drive] += size
        self._stats_total = total
        self._stats_by_drive = sorted(
            by_drive.items(), key=lambda item: (-item[1], item[0])
        )

    def _get_drive_label(self, path):
        drive, _ = os.path.splitdrive(path)
        if drive:
            return drive.rstrip(":").upper()
        if path.startswith("\\\\"):
            return "UNC"
        return "ROOT"

    def _get_model_types(self):
        types = []
        if isinstance(getattr(self, "models_def", None), dict) and self.models_def:
            types = sorted(self.models_def.keys())
        elif isinstance(self.transformer_types, list) and self.transformer_types:
            types = list(self.transformer_types)
        elif isinstance(self.displayed_model_types, list):
            types = list(self.displayed_model_types)
        seen = set()
        ordered = []
        for model_type in types:
            if model_type and model_type not in seen:
                ordered.append(model_type)
                seen.add(model_type)
        return ordered

    def _collect_model_variants(self, model_type):
        model_def = self.get_model_def(model_type)
        if model_def is None:
            return {"variants": [], "primary_paths": [], "files": set()}

        default_dtype_policy = self._resolve_default_dtype_policy(model_type, model_def)
        transformer_variants = self._collect_transformer_variants(
            model_type, model_def, default_dtype_policy
        )
        text_encoder_variants = self._collect_text_encoder_variants(
            model_type, model_def, default_dtype_policy
        )
        transformer_files = set()
        text_encoder_files = set()
        model_primary_paths = []
        for variant in transformer_variants:
            transformer_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        for variant in text_encoder_variants:
            text_encoder_files.update(variant["files"])
            model_primary_paths.extend(variant.get("primary_paths", []))
        model_primary_paths = self._dedupe_paths(model_primary_paths)

        other_files, shared_files = self._collect_other_files(
            model_type,
            model_def,
            transformer_files,
            text_encoder_files,
        )
        variants = list(transformer_variants) + list(text_encoder_variants)
        if other_files:
            other_id = self._make_variant_id(model_type, "other", "other")
            variants.append(
                {
                    "id": other_id,
                    "label": "Other",
                    "type": "variant",
                    "files": other_files,
                    "primary_paths": [],
                }
            )
        if shared_files:
            shared_id = self._make_variant_id(model_type, "shared", "shared")
            variants.append(
                {
                    "id": shared_id,
                    "label": "Shared",
                    "type": "variant",
                    "files": shared_files,
                    "primary_paths": [],
                }
            )

        model_files = set()
        for variant in variants:
            model_files.update(variant.get("files", set()))

        return {
            "variants": variants,
            "primary_paths": model_primary_paths,
            "files": model_files,
        }

    def _collect_transformer_variants(
        self, model_type, model_def, default_dtype_policy
    ):
        choice_lists, modules = self._collect_transformer_choice_lists(model_type, model_def)
        base_dtypes, token_dtypes, token_labels = self._collect_variant_candidates(
            choice_lists, default_dtype_policy
        )
        variants = []
        used_labels = set()

        base_default = self._format_dtype_label(default_dtype_policy)
        if base_dtypes:
            for dtype_policy in sorted(base_dtypes):
                label = self._format_variant_label("", dtype_policy, base_default)
                files, primary_paths = self._collect_transformer_files_for_variant(
                    model_type, model_def, modules, "", dtype_policy
                )
                if not files:
                    continue
                file_label = self._detect_quant_label_from_files(files)
                label = file_label or label
                variant_label = f"Transformer {label}"
                variant_id = self._make_variant_id(model_type, "transformer", label)
                variants.append(
                    {
                        "id": variant_id,
                        "label": variant_label,
                        "type": "variant",
                        "files": files,
                        "primary_paths": primary_paths,
                    }
                )
                used_labels.add(variant_label)

        for token, dtypes in sorted(token_dtypes.items()):
            quant_param = self._quant_param_from_token(token)
            quant_label = token_labels.get(token) or self._format_quant_label(token)
            for dtype_policy in sorted(dtypes):
                label_suffix = ""
                if len(dtypes) > 1:
                    label_suffix = f" {self._format_dtype_label(dtype_policy)}"
                label = f"{quant_label}{label_suffix}"
                files, primary_paths = self._collect_transformer_files_for_variant(
                    model_type,
                    model_def,
                    modules,
                    quant_param,
                    dtype_policy,
                    token_match=token,
                )
                if not files:
                    continue
                file_label = self._detect_quant_label_from_files(files)
                quant_label = file_label or quant_label
                if not self._files_match_token(files, token):
                    continue
                label = f"{quant_label}{label_suffix}" if quant_label else self._format_dtype_label(dtype_policy)
                variant_label = f"Transformer {label}"
                if variant_label in used_labels:
                    continue
                variant_id = self._make_variant_id(model_type, "transformer", label)
                variants.append(
                    {
                        "id": variant_id,
                        "label": variant_label,
                        "type": "variant",
                        "files": files,
                        "primary_paths": primary_paths,
                    }
                )
                used_labels.add(variant_label)

        return sorted(variants, key=lambda v: v["label"].lower())

    def _collect_transformer_choice_lists(self, model_type, model_def):
        choice_lists = []
        modules = self._expand_modules(model_type)

        urls = self.get_model_recursive_prop(model_type, "URLs", return_list=True)
        if urls:
            choice_lists.append(urls)
        if "URLs2" in model_def:
            urls2 = self.get_model_recursive_prop(model_type, "URLs2", return_list=True)
            if urls2:
                choice_lists.append(urls2)

        has_module_source = "module_source" in model_def
        has_module_source2 = "module_source2" in model_def
        for module in modules:
            if isinstance(module, dict):
                urls1 = module.get("URLs", [])
                urls2 = module.get("URLs2", [])
                if urls1 and not has_module_source:
                    choice_lists.append(urls1)
                if urls2 and not has_module_source2:
                    choice_lists.append(urls2)
            elif isinstance(module, list):
                if has_module_source:
                    continue
                if module:
                    choice_lists.append(module)
            else:
                if has_module_source:
                    continue
                module_type = module
                sub_prop_name = "_list"
                if isinstance(module_type, str) and "#" in module_type:
                    pos = module_type.rfind("#")
                    sub_prop_name = module_type[pos + 1 :]
                    module_type = module_type[:pos]
                try:
                    mod_urls = self.get_model_recursive_prop(
                        module_type, "modules", sub_prop_name=sub_prop_name, return_list=True
                    )
                except Exception:
                    mod_urls = []
                if isinstance(mod_urls, list) and mod_urls:
                    if all(isinstance(item, list) for item in mod_urls):
                        for entry in mod_urls:
                            if entry:
                                choice_lists.append(entry)
                    else:
                        choice_lists.append(mod_urls)
        return choice_lists, modules

    def _collect_transformer_files_for_variant(
        self,
        model_type,
        model_def,
        modules,
        quantization,
        dtype_policy,
        token_match=None,
    ):
        files = set()
        primary_paths = []

        model_filename = ""
        is_exotic = quantization not in ("", "int8", "fp8")
        if token_match and is_exotic:
            urls = self.get_model_recursive_prop(model_type, "URLs", return_list=True)
            model_filename = self._select_filename_by_token(urls, token_match)
        if not model_filename:
            model_filename = self.get_model_filename(
                model_type=model_type,
                quantization=quantization,
                dtype_policy=dtype_policy,
            )
        if model_filename:
            self._add_file(files, model_filename)
            primary_path = self._resolve_path(model_filename)
            if primary_path and primary_path in files:
                primary_paths.append(primary_path)

        if "URLs2" in model_def:
            model_filename2 = ""
            if token_match and is_exotic:
                urls2 = self.get_model_recursive_prop(
                    model_type, "URLs2", return_list=True
                )
                model_filename2 = self._select_filename_by_token(urls2, token_match)
            if not model_filename2:
                model_filename2 = self.get_model_filename(
                    model_type=model_type,
                    quantization=quantization,
                    dtype_policy=dtype_policy,
                    submodel_no=2,
                )
            if model_filename2:
                self._add_file(files, model_filename2)
                primary_path2 = self._resolve_path(model_filename2)
                if primary_path2 and primary_path2 in files:
                    primary_paths.append(primary_path2)

        has_module_source = "module_source" in model_def
        has_module_source2 = "module_source2" in model_def
        for module in modules:
            if isinstance(module, dict):
                urls1 = module.get("URLs", [])
                urls2 = module.get("URLs2", [])
                if urls1 and not has_module_source:
                    filename = ""
                    if token_match and is_exotic:
                        filename = self._select_filename_by_token(urls1, token_match)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            URLs=urls1,
                        )
                    if filename:
                        self._add_file(files, filename)
                if urls2 and not has_module_source2:
                    filename = ""
                    if token_match and is_exotic:
                        filename = self._select_filename_by_token(urls2, token_match)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            URLs=urls2,
                        )
                    if filename:
                        self._add_file(files, filename)
            elif isinstance(module, list):
                if has_module_source:
                    continue
                filename = ""
                if token_match and is_exotic:
                    filename = self._select_filename_by_token(module, token_match)
                if not filename:
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization=quantization,
                        dtype_policy=dtype_policy,
                        URLs=module,
                    )
                if filename:
                    self._add_file(files, filename)
            else:
                if has_module_source:
                    continue
                if token_match and is_exotic:
                    module_type = module
                    sub_prop_name = "_list"
                    if isinstance(module_type, str) and "#" in module_type:
                        pos = module_type.rfind("#")
                        sub_prop_name = module_type[pos + 1 :]
                        module_type = module_type[:pos]
                    try:
                        mod_urls = self.get_model_recursive_prop(
                            module_type,
                            "modules",
                            sub_prop_name=sub_prop_name,
                            return_list=True,
                        )
                    except Exception:
                        mod_urls = []
                    filename = self._select_filename_by_token(mod_urls, token_match)
                else:
                    filename = ""
                try:
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quantization,
                            dtype_policy=dtype_policy,
                            module_type=module,
                        )
                except Exception:
                    filename = ""
                if filename:
                    self._add_file(files, filename)

        return files, primary_paths

    def _collect_text_encoder_variants(
        self, model_type, model_def, default_dtype_policy
    ):
        variants = []
        text_encoder_urls = self.get_model_recursive_prop(
            model_type, "text_encoder_URLs", return_list=True
        )
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        used_paths = set()
        used_labels = set()
        base_default = self._format_dtype_label(default_dtype_policy)
        if text_encoder_urls:
            base_dtypes, token_dtypes, token_labels = self._collect_variant_candidates(
                [text_encoder_urls], default_dtype_policy
            )
            if base_dtypes:
                for dtype_policy in sorted(base_dtypes):
                    label = self._format_variant_label("", dtype_policy, base_default)
                    filename = self.get_model_filename(
                        model_type=model_type,
                        quantization="",
                        dtype_policy=dtype_policy,
                        URLs=text_encoder_urls,
                    )
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)

            for token, dtypes in sorted(token_dtypes.items()):
                quant_param = self._quant_param_from_token(token)
                quant_label = token_labels.get(token) or self._format_quant_label(token)
                for dtype_policy in sorted(dtypes):
                    label_suffix = ""
                    if len(dtypes) > 1:
                        label_suffix = f" {self._format_dtype_label(dtype_policy)}"
                    label = f"{quant_label}{label_suffix}"
                    filename = ""
                    if quant_param not in ("", "int8", "fp8"):
                        filename = self._select_filename_by_token(text_encoder_urls, token)
                    if not filename:
                        filename = self.get_model_filename(
                            model_type=model_type,
                            quantization=quant_param,
                            dtype_policy=dtype_policy,
                            URLs=text_encoder_urls,
                        )
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    if not self._files_match_token(files, token):
                        continue
                    file_label = self._detect_quant_label_from_files(files)
                    quant_label = file_label or quant_label
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    label = f"{quant_label}{label_suffix}" if quant_label else self._format_dtype_label(dtype_policy)
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)
        else:
            handler = self.get_model_handler(model_type)
            get_name = getattr(handler, "get_text_encoder_filename", None)
            if get_name is not None:
                for quant_param in self._text_encoder_quant_candidates():
                    try:
                        filename = get_name(quant_param)
                    except Exception:
                        filename = None
                    if not filename:
                        continue
                    files = set()
                    self._add_file(
                        files, filename, force_folder=text_encoder_folder
                    )
                    if not files:
                        continue
                    resolved_path = next(iter(files))
                    if resolved_path in used_paths:
                        continue
                    used_paths.add(resolved_path)
                    quant_label = self._detect_quant_label_from_files(files)
                    label = quant_label or self._format_variant_label(
                        "",
                        self._detect_dtype_policy_from_name(os.path.basename(filename))
                        or default_dtype_policy,
                        base_default,
                    )
                    variant_label = f"Text Encoder {label}"
                    if variant_label in used_labels:
                        continue
                    variant_id = self._make_variant_id(model_type, "text", label)
                    variants.append(
                        {
                            "id": variant_id,
                            "label": variant_label,
                            "type": "variant",
                            "files": files,
                            "primary_paths": [resolved_path],
                        }
                    )
                    used_labels.add(variant_label)

        return sorted(variants, key=lambda v: v["label"].lower())

    def _collect_other_files(
        self,
        model_type,
        model_def,
        transformer_files,
        text_encoder_files,
    ):
        other_files = set()
        shared_files = set()

        preload_urls = self.get_model_recursive_prop(
            model_type, "preload_URLs", return_list=True
        )
        for url in self._ensure_list(preload_urls):
            self._add_file(other_files, url)

        vae_urls = model_def.get("VAE_URLs", [])
        for url in self._ensure_list(vae_urls):
            self._add_file(shared_files, url)

        loras = self.get_model_recursive_prop(model_type, "loras", return_list=True)
        for url in self._ensure_list(loras):
            lora_dir = self._safe_get_lora_dir(model_type)
            if lora_dir:
                lora_path = os.path.join(lora_dir, os.path.basename(url))
                self._add_file(other_files, lora_path)

        handler_files = self._collect_handler_files(model_type, model_def)
        shared_files.update(handler_files)

        other_files.difference_update(transformer_files)
        other_files.difference_update(text_encoder_files)
        shared_files.difference_update(transformer_files)
        shared_files.difference_update(text_encoder_files)
        shared_files.difference_update(other_files)
        return other_files, shared_files

    def _resolve_default_dtype_policy(self, model_type, model_def):
        dtype = model_def.get("dtype")
        if isinstance(dtype, str) and dtype.lower() in ("fp16", "bf16"):
            return dtype.lower()
        policy = self.transformer_dtype_policy
        if isinstance(policy, str) and policy:
            return policy.lower()
        return "bf16"

    def _format_dtype_label(self, dtype_policy):
        return "FP16" if dtype_policy == "fp16" else "BF16"

    def _format_quant_label(self, token):
        if not token:
            return ""
        label = quant_router.get_quantization_label(token)
        if label:
            return label
        return str(token).replace("_", " ").upper()

    def _format_variant_label(self, token, dtype_policy, default_label):
        if token:
            return self._format_quant_label(token)
        if dtype_policy:
            return self._format_dtype_label(dtype_policy)
        return default_label

    def _detect_quant_info(self, name):
        if not name:
            return "", ""
        label = quant_router.detect_quantization_label_from_filename(name) or ""
        if label:
            return self._label_to_token(label), label
        kind = quant_router.detect_quantization_kind_for_file(name, verboseLevel=0)
        if kind and kind != "none":
            kind = str(kind).lower()
            return kind, kind.upper()
        lower = str(name).lower()
        aliases = self._get_quant_aliases()
        for token in aliases:
            if token in ("bf16", "fp16", "bfloat16", "float16"):
                continue
            if token and token in lower:
                return token, ""
        return "", ""

    def _label_to_token(self, label):
        if not label:
            return ""
        raw = str(label).strip()
        lower = raw.lower()
        if lower.startswith("gguf-"):
            return lower.split("-", 1)[1]
        if lower == "gguf":
            return "gguf"
        return lower

    def _detect_dtype_policy_from_name(self, name):
        if not name:
            return ""
        lower = str(name).lower()
        if "fp16" in lower or "float16" in lower:
            return "fp16"
        if "bf16" in lower or "bfloat16" in lower:
            return "bf16"
        return ""

    def _collect_variant_candidates(self, choice_lists, default_dtype_policy):
        base_dtypes = set()
        token_dtypes = defaultdict(set)
        token_labels = {}
        for choice_list in choice_lists:
            for entry in self._ensure_list(choice_list):
                if not entry:
                    continue
                entry_str = str(entry)
                name = os.path.basename(entry_str)
                quant_source = self._resolve_path(entry_str) or entry_str
                token, label = self._detect_quant_info(quant_source)
                dtype_policy = self._detect_dtype_policy_from_name(name) or default_dtype_policy
                if token:
                    token = str(token).lower()
                    token_dtypes[token].add(dtype_policy)
                    if label:
                        token_labels.setdefault(token, label)
                else:
                    base_dtypes.add(dtype_policy)
        if not base_dtypes and not token_dtypes:
            base_dtypes.add(default_dtype_policy)
        return base_dtypes, token_dtypes, token_labels

    def _quant_param_from_token(self, token):
        if not token:
            return ""
        lower = str(token).lower()
        if "int8" in lower:
            return "int8"
        if "fp8" in lower or "float8" in lower:
            return "fp8"
        return lower

    def _files_match_token(self, files, token):
        if not token:
            return True
        token = str(token).lower()
        for path in files:
            base = os.path.basename(path).lower()
            if token in base:
                return True
            label = quant_router.detect_quantization_label_from_filename(path)
            if label and self._label_to_token(label) == token:
                return True
        return False

    def _detect_quant_label_from_files(self, files):
        for path in files:
            label = quant_router.detect_quantization_label_from_filename(path)
            if label:
                return label
        return ""

    def _select_filename_by_token(self, urls, token):
        if not urls or not token:
            return ""
        token = str(token).lower()
        for entry in self._ensure_list(urls):
            if not entry:
                continue
            if isinstance(entry, list):
                for item in entry:
                    if not item:
                        continue
                    base = os.path.basename(str(item)).lower()
                    if token in base:
                        return item
                    label = quant_router.detect_quantization_label_from_filename(item)
                    if label and self._label_to_token(label) == token:
                        return item
                continue
            base = os.path.basename(str(entry)).lower()
            if token in base:
                return entry
            label = quant_router.detect_quantization_label_from_filename(entry)
            if label and self._label_to_token(label) == token:
                return entry
        return ""

    def _expand_modules(self, model_type):
        modules = self.get_model_recursive_prop(model_type, "modules", return_list=True)
        expanded_modules = []
        for module in modules:
            if isinstance(module, str):
                if "#" in module:
                    expanded_modules.append(module)
                    continue
                try:
                    expanded = self.get_model_recursive_prop(
                        module, "modules", sub_prop_name="_list", return_list=True
                    )
                except Exception:
                    expanded = []
                if expanded:
                    expanded_modules.append(expanded)
                else:
                    expanded_modules.append(module)
            else:
                expanded_modules.append(module)
        return expanded_modules

    def _make_variant_id(self, model_type, kind, label):
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        return f"variant::{model_type}::{kind}::{slug}"

    def _dedupe_paths(self, paths):
        seen = set()
        ordered = []
        for path in paths:
            if not path or path in seen:
                continue
            ordered.append(path)
            seen.add(path)
        return ordered

    def _get_quant_aliases(self):
        if not hasattr(self, "_quant_alias_cache"):
            aliases = set(quant_router.get_available_qtype_aliases() or [])
            aliases.update({"int8", "int8n", "fp8"})
            self._quant_alias_cache = sorted(
                {alias for alias in aliases if alias}, key=len, reverse=True
            )
        return self._quant_alias_cache

    def _text_encoder_quant_candidates(self):
        candidates = {""}
        aliases = self._get_quant_aliases()
        for token in aliases:
            if "int8" in token:
                candidates.add("int8")
            if "fp8" in token or "float8" in token:
                candidates.add("fp8")
        return sorted(candidates)

    def _collect_text_encoder_files(self, model_type, model_def):
        files = set()
        text_encoder_urls = self.get_model_recursive_prop(
            model_type, "text_encoder_URLs", return_list=True
        )
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        text_encoder_filename = None
        if text_encoder_urls:
            text_encoder_filename = self.get_model_filename(
                model_type=model_type,
                quantization=self.text_encoder_quantization,
                dtype_policy=self.transformer_dtype_policy,
                URLs=text_encoder_urls,
            )
        else:
            handler = self.get_model_handler(model_type)
            get_name = getattr(handler, "get_text_encoder_filename", None)
            if get_name is not None:
                try:
                    text_encoder_filename = get_name(self.text_encoder_quantization)
                except Exception:
                    text_encoder_filename = None
        if text_encoder_filename:
            self._add_file(files, text_encoder_filename, force_folder=text_encoder_folder)
        return files

    def _collect_handler_files(self, model_type, model_def):
        handler = self.get_model_handler(model_type)
        query = getattr(handler, "query_model_files", None)
        if query is None:
            return set()
        base_model_type = self.get_base_model_type(model_type)
        try:
            download_defs = query(
                self._compute_list,
                base_model_type,
                model_def,
            )
        except Exception as exc:
            self._errors.append(f"{model_type}: {exc}")
            return set()

        if not download_defs:
            return set()
        if isinstance(download_defs, dict):
            download_defs = [download_defs]

        files = set()
        for download_def in download_defs:
            source_folders = download_def.get("sourceFolderList", [])
            file_lists = download_def.get("fileList", [])
            target_folders = download_def.get("targetFolderList")
            if target_folders is None:
                target_folders = [None] * len(source_folders)

            for source_folder, file_list, target_folder in zip(
                source_folders, file_lists, target_folders
            ):
                if target_folder == "":
                    target_folder = None
                if source_folder is None:
                    source_folder = ""

                if not file_list:
                    continue

                for filename in file_list:
                    if not filename:
                        continue
                    rel_path = self._combine_download_path(
                        target_folder, source_folder, filename
                    )
                    files.add(self._resolve_download_relpath(rel_path))
        return {path for path in files if path and os.path.isfile(path)}

    def _build_tree_structure(self, model_types):
        families = defaultdict(list)
        for model_type in model_types:
            family = self.get_model_family(model_type, for_ui=True) or "unknown"
            families[family].append(model_type)

        family_order = sorted(
            families.keys(),
            key=lambda family: self.families_infos.get(family, (999, family))[0],
        )

        for family in family_order:
            family_label = self.families_infos.get(family, (999, family))[1]
            rows = []
            for model_type in families[family]:
                model_name = self.compact_name(
                    family_label, self.get_model_name(model_type)
                )
                parent_id = self.get_parent_model_type(model_type)
                rows.append((model_name, model_type, parent_id))

            rows.sort(key=lambda row: row[0])
            parents_list, children_dict = self.create_models_hierarchy(rows)
            family_node_id = f"family::{family}"
            family_models = set(families[family])
            family_usage_ids = self._collect_usage_ids_for_models(family_models)
            family_files = self._collect_files_for_usage_ids(family_usage_ids)
            self._register_node(
                family_node_id,
                family_label,
                "family",
                family_models,
                family_files,
                path_label=family_label,
            )

            base_nodes = []
            for base_label, base_id in parents_list:
                child_entries = children_dict.get(base_id, [])
                child_nodes = []
                base_models = set()
                base_usage_ids = set()
                base_path = f"{family_label} / {base_label}"
                for child_label, child_model in child_entries:
                    base_models.add(child_model)
                    model_usage_ids = self._model_usage_ids.get(child_model, set())
                    base_usage_ids.update(model_usage_ids)
                    self._model_base_label[child_model] = base_label
                    self._model_family_label[child_model] = family_label
                    model_node_id = f"model::{child_model}"
                    model_files = self._collect_files_for_usage_ids(model_usage_ids)
                    primary_paths = self._model_primary.get(child_model, [])
                    model_path = f"{base_path} / {child_label}"
                    self._register_node(
                        model_node_id,
                        child_label,
                        "model",
                        {child_model},
                        model_files,
                        primary_paths=primary_paths,
                        path_label=model_path,
                    )
                    variant_children = []
                    for variant in self._model_variants.get(child_model, []):
                        variant_id = variant["id"]
                        variant_label = variant["label"]
                        variant_path = f"{model_path} / {variant_label}"
                        self._register_node(
                            variant_id,
                            variant_label,
                            "variant",
                            {child_model},
                            variant.get("files", set()),
                            primary_paths=variant.get("primary_paths", []),
                            path_label=variant_path,
                        )
                        variant_children.append(
                            {"id": variant_id, "label": variant_label, "type": "variant"}
                        )
                    child_nodes.append(
                        {
                            "id": model_node_id,
                            "label": child_label,
                            "type": "model",
                            "children": variant_children,
                        }
                    )

                base_node_id = f"base::{family}::{base_id}"
                base_files = self._collect_files_for_usage_ids(base_usage_ids)
                self._register_node(
                    base_node_id,
                    base_label,
                    "base",
                    base_models,
                    base_files,
                    path_label=base_path,
                )
                base_nodes.append(
                    {
                        "id": base_node_id,
                        "label": base_label,
                        "type": "base",
                        "children": child_nodes,
                    }
                )

            self._tree_data.append(
                {
                    "id": family_node_id,
                    "label": family_label,
                    "type": "family",
                    "children": base_nodes,
                }
            )

    def _collect_usage_ids_for_models(self, model_set):
        usage_ids = set()
        for model_type in model_set:
            usage_ids.update(self._model_usage_ids.get(model_type, set()))
        return usage_ids

    def _collect_files_for_usage_ids(self, usage_ids):
        files = set()
        for usage_id in usage_ids:
            files.update(self._usage_files.get(usage_id, set()))
        return files

    def _register_node(
        self,
        node_id,
        label,
        node_type,
        model_set,
        files_set,
        primary_paths=None,
        path_label=None,
    ):
        unique_files, shared_files = self._split_files_by_shared(files_set, model_set)
        unique_size = self._sum_sizes(unique_files)
        shared_size = self._sum_sizes(shared_files)
        self._node_map[node_id] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "models": set(model_set),
            "files": set(files_set),
            "unique_files": unique_files,
            "shared_files": shared_files,
            "unique_size": unique_size,
            "shared_size": shared_size,
            "total_size": unique_size + shared_size,
            "primary_paths": primary_paths or [],
            "path_label": path_label or label,
        }

    def _sum_sizes(self, files_set):
        total = 0
        for path in files_set:
            total += self._file_info.get(path, {}).get("size", 0)
        return total

    def _split_files_by_shared(self, files_set, model_set):
        unique_files = set()
        shared_files = set()
        for path in files_set:
            used_by = self._file_usage_models.get(path, set())
            if used_by.issubset(model_set):
                unique_files.add(path)
            else:
                shared_files.add(path)
        return unique_files, shared_files

    def _rebalance_other_shared_variants(self, model_types):
        for model_type in model_types:
            variants = self._model_variants.get(model_type, [])
            if not variants:
                continue
            other_variant = None
            shared_variant = None
            for variant in variants:
                label = str(variant.get("label", "")).lower()
                if label == "other":
                    other_variant = variant
                elif label == "shared":
                    shared_variant = variant

            if not other_variant and not shared_variant:
                continue

            combined = set()
            if other_variant:
                combined.update(other_variant.get("files", set()))
            if shared_variant:
                combined.update(shared_variant.get("files", set()))

            new_variants = []
            for variant in variants:
                if variant is other_variant or variant is shared_variant:
                    continue
                new_variants.append(variant)
            if not combined:
                self._model_variants[model_type] = new_variants
                continue

            model_set = {model_type}
            shared_files = {
                path
                for path in combined
                if not self._file_usage_models.get(path, set()).issubset(model_set)
            }
            other_files = combined - shared_files

            if other_files:
                if other_variant is None:
                    other_variant = {
                        "id": self._make_variant_id(model_type, "other", "other"),
                        "label": "Other",
                        "type": "variant",
                        "files": other_files,
                        "primary_paths": [],
                    }
                else:
                    other_variant["files"] = other_files
                new_variants.append(other_variant)

            if shared_files:
                if shared_variant is None:
                    shared_variant = {
                        "id": self._make_variant_id(model_type, "shared", "shared"),
                        "label": "Shared",
                        "type": "variant",
                        "files": shared_files,
                        "primary_paths": [],
                    }
                else:
                    shared_variant["files"] = shared_files
                new_variants.append(shared_variant)

            self._model_variants[model_type] = new_variants

    def _shared_base_labels(self, path, exclude_key=None):
        pairs = set()
        for model_type in self._file_usage_models.get(path, set()):
            base_label = self._model_base_label.get(model_type)
            family_label = self._model_family_label.get(model_type)
            if base_label and family_label:
                pairs.add((family_label, base_label))
        if exclude_key:
            pairs.discard(exclude_key)
        return [
            f"{family} {base}"
            for family, base in sorted(pairs, key=lambda item: (item[0].lower(), item[1].lower()))
        ]

    def _get_node_base_label(self, node_id, node_type):
        model_type = None
        if node_type == "base":
            family_label = None
            if node_id.startswith("base::"):
                parts = node_id.split("::", 2)
                if len(parts) >= 2:
                    family_key = parts[1]
                    family_label = self.families_infos.get(
                        family_key, (999, family_key)
                    )[1]
            info = self._node_map.get(node_id, {})
            base_label = info.get("label")
            if family_label and base_label:
                return (family_label, base_label)
            return None
        if node_type == "variant":
            model_type = self._variant_to_model.get(node_id)
        elif node_type == "model":
            if node_id.startswith("model::"):
                model_type = node_id.split("model::", 1)[1]
        if model_type:
            family_label = self._model_family_label.get(model_type)
            base_label = self._model_base_label.get(model_type)
            if family_label and base_label:
                return (family_label, base_label)
        return None

    def _build_tree_html(self):
        css = """
        <style>
        .ckpt-wrap { display: flex; flex-direction: column; gap: 12px; }
        .ckpt-panel { border: 1px solid var(--border-color-primary); border-radius: 14px; background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02)); box-shadow: var(--shadow-drop-sm, 0 2px 8px rgba(0,0,0,0.08)); }
        .ckpt-panel-inner { padding: 10px; }
        .ckpt-tree { display: flex; flex-direction: column; gap: 8px; }
        .ckpt-node { border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--background-fill-secondary); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
        .ckpt-node summary { list-style: none; cursor: pointer; padding: 8px 10px; }
        .ckpt-node summary::-webkit-details-marker { display: none; }
        .ckpt-row { display: flex; align-items: center; gap: 10px; border-radius: 8px; padding: 6px 6px; }
        .ckpt-row.clickable { cursor: pointer; transition: background 0.15s ease; }
        .ckpt-row.clickable:hover { background: rgba(127,127,127,0.08); }
        .ckpt-label { flex: 1; font-weight: 600; color: inherit; }
        .ckpt-sizes { display: flex; flex-direction: column; align-items: stretch; min-width: 128px; width: 128px; font-size: 0.85em; font-variant-numeric: tabular-nums; gap: 4px; }
        .ckpt-size-unique { color: inherit; }
        .ckpt-size-shared { color: inherit; }
        .ckpt-badge { padding: 3px 8px; border-radius: 999px; font-weight: 600; font-size: 0.82em; font-variant-numeric: tabular-nums; min-width: 84px; width: 100%; display: inline-flex; align-items: center; justify-content: center; }
        .ckpt-badge-unique { background: rgba(64, 120, 255, 0.18); color: #2f5bd4; border: 1px solid rgba(64, 120, 255, 0.4); }
        .ckpt-badge-shared { background: rgba(240, 200, 40, 0.25); color: #a66a00; border: 1px solid rgba(240, 200, 40, 0.5); }
        .ckpt-actions { display: flex; gap: 18px; }
        .ckpt-action-btn { padding: 3px 12px; height: 30px; border: 1px solid var(--border-color-primary); border-radius: 6px; cursor: pointer; font-size: 0.85em; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; background: transparent; color: inherit; }
        .ckpt-action-btn:hover { box-shadow: var(--shadow-drop, 0 2px 6px rgba(0,0,0,0.12)); }
        .ckpt-view-btn { background: rgba(60, 120, 255, 0.9) !important; color: #ffffff !important; border-color: rgba(60, 120, 255, 0.95) !important; }
        .ckpt-view-btn:hover { background: rgba(60, 120, 255, 1) !important; }
        .ckpt-delete { background: rgba(220, 60, 60, 0.95) !important; color: #ffffff !important; border-color: rgba(220, 60, 60, 0.95) !important; }
        .ckpt-delete:hover { background: rgba(220, 60, 60, 1) !important; }
        .ckpt-children { padding: 6px 8px 10px 18px; display: flex; flex-direction: column; gap: 6px; }
        .ckpt-leaf { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); padding: 6px 8px; }
        .ckpt-right-column { align-self: flex-start; position: relative; overflow: visible !important; }
        .ckpt-view-sticky { position: sticky; top: 16px; align-self: flex-start; }
        .ckpt-view { border: 1px solid var(--border-color-primary); border-radius: 12px; background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02)); padding: 12px; max-height: none; overflow: visible; }
        .ckpt-view h4 { margin: 0 0 6px 0; }
        .ckpt-view-summary { font-size: 0.85em; color: inherit; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 12px; }
        .ckpt-loading-message { font-size: 1.05em; font-weight: 600; color: inherit; padding: 16px 8px; }
        .ckpt-muted { opacity: 0.75; }
        .ckpt-stats { display: flex; flex-direction: column; gap: 6px; }
        .ckpt-stat-main { font-weight: 600; font-size: 0.95em; }
        .ckpt-drive-list { display: flex; flex-wrap: wrap; gap: 8px; }
        .ckpt-drive { padding: 4px 8px; border-radius: 999px; background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); font-size: 0.82em; }
        .ckpt-file-tree { display: flex; flex-direction: column; gap: 6px; max-height: none; overflow: visible; }
        .ckpt-folder { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); }
        .ckpt-folder.shared { border-color: rgba(240,200,40,0.45); }
        .ckpt-folder.shared > summary { background: rgba(240,200,40,0.12); }
        .ckpt-folder.unique { border-color: rgba(64, 120, 255, 0.35); }
        .ckpt-folder.unique > summary { background: rgba(64, 120, 255, 0.12); }
        .ckpt-folder summary { list-style: none; cursor: pointer; padding: 6px 8px; display: flex; align-items: center; gap: 8px; position: relative; }
        .ckpt-folder summary::-webkit-details-marker { display: none; }
        .ckpt-folder-name { flex: 1; font-weight: 600; }
        .ckpt-folder-size { font-size: 0.8em; color: inherit; opacity: 0.75; font-variant-numeric: tabular-nums; }
        .ckpt-folder-children { padding: 6px 8px 8px 18px; display: flex; flex-direction: column; gap: 4px; }
        .ckpt-file-row { display: flex; gap: 8px; align-items: center; font-size: 0.85em; padding: 2px 0; }
        .ckpt-file-row.shared { position: relative; }
        .ckpt-file-row.shared { color: #a66a00; background: rgba(240, 200, 40, 0.12); border-radius: 6px; padding: 2px 6px; }
        .ckpt-file-row.unique { background: rgba(64, 120, 255, 0.12); border-radius: 6px; padding: 2px 6px; }
        .ckpt-file-row.missing { opacity: 0.6; }
        .ckpt-file-name { flex: 1; word-break: break-all; }
        .ckpt-file-size { min-width: 70px; text-align: right; color: inherit; opacity: 0.75; font-variant-numeric: tabular-nums; }
        .ckpt-shared-tag { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.04em; border: 1px solid rgba(240,200,40,0.5); color: #a66a00; padding: 2px 6px; border-radius: 999px; background: rgba(240,200,40,0.12); }
        .ckpt-folder-tag { margin-left: 6px; }
        .ckpt-shared-tooltip { opacity: 0; visibility: hidden; pointer-events: none; position: absolute; top: 100%; left: 0; margin-top: 6px; padding: 6px 8px; background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 8px; box-shadow: var(--shadow-drop, 0 2px 6px rgba(0,0,0,0.12)); font-size: 0.8em; z-index: 10; min-width: 160px; transform: translateY(4px); transition: opacity 0.15s ease, transform 0.15s ease; transition-delay: 0s; }
        .ckpt-shared-tooltip-title { font-weight: 600; margin-bottom: 4px; }
        .ckpt-file-row.shared:hover .ckpt-shared-tooltip { opacity: 1; visibility: visible; transform: translateY(0); transition-delay: 2s; }
        .ckpt-folder.shared summary:hover .ckpt-shared-tooltip { opacity: 1; visibility: visible; transform: translateY(0); transition-delay: 2s; }
        .ckpt-path { font-size: 0.82em; color: inherit; opacity: 0.8; margin-bottom: 6px; word-break: break-all; }
        .ckpt-primary { font-size: 0.82em; color: inherit; opacity: 0.8; margin-bottom: 8px; word-break: break-all; }
        .ckpt-errors { padding: 8px 10px; border: 1px solid rgba(220, 80, 80, 0.4); background: var(--background-fill-primary); border-radius: 8px; font-size: 0.85em; }
        .ckpt-modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.45); display: none; align-items: center; justify-content: center; z-index: 9999; }
        .ckpt-modal { width: min(420px, 90vw); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px; box-shadow: 0 12px 28px rgba(0,0,0,0.25); display: flex; flex-direction: column; gap: 12px; }
        .ckpt-modal-title { font-weight: 700; font-size: 1.05em; }
        .ckpt-modal-body { font-size: 0.95em; color: inherit; opacity: 0.8; }
        .ckpt-modal-actions { display: flex; gap: 12px; justify-content: flex-end; }
        .ckpt-modal-btn { padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: var(--button-secondary-background-fill, #f0f0f0); cursor: pointer; font-weight: 600; }
        .ckpt-modal-btn.danger { background: rgba(220,80,80,0.15); border-color: rgba(220,80,80,0.5); color: #c62828; }
        </style>
        """

        tree_parts = [
            css,
            "<div id='ckpt_tree_root' data-status='ready' class='ckpt-wrap'>",
        ]
        if self._errors:
            errors_html = "<br>".join(html.escape(err) for err in self._errors[:10])
            tree_parts.append(f"<div class='ckpt-errors'>{errors_html}</div>")

        tree_parts.append("<div class='ckpt-panel'><div class='ckpt-panel-inner'>")
        tree_parts.append(self._build_stats_html())
        tree_parts.append("<div class='ckpt-tree'>")
        for family in self._tree_data:
            tree_parts.append(self._render_node(family, is_family=True))
        tree_parts.append("</div></div></div></div>")
        return "".join(tree_parts)

    def _build_stats_html(self):
        total_label = self._format_gb(self._stats_total)
        drive_html = ""
        if len(self._stats_by_drive) > 1:
            drive_items = []
            for drive, size in self._stats_by_drive:
                drive_items.append(
                    f"<span class='ckpt-drive'>{html.escape(drive)}: {self._format_gb(size)}</span>"
                )
            drive_html = f"<div class='ckpt-drive-list'>{''.join(drive_items)}</div>"
        return (
            "<div class='ckpt-stats'>"
            f"<div class='ckpt-stat-main'>Total Used: {total_label}</div>"
            f"{drive_html}</div>"
        )

    def _render_node(self, node, is_family=False):
        node_id = node["id"]
        info = self._node_map.get(node_id, {})
        label = info.get("label", node.get("label", node_id))
        unique_size_value = info.get("unique_size", 0)
        shared_size_value = info.get("shared_size", 0)
        unique = self._format_gb(unique_size_value)
        shared = self._format_gb(shared_size_value)
        unique_label = self._format_size(unique_size_value)
        shared_label = self._format_size(shared_size_value)
        total_label = self._format_size(unique_size_value + shared_size_value)
        node_type = info.get("type", node.get("type", ""))

        children = node.get("children", [])
        click_view = node_type in ("model", "variant")
        row_html = self._render_row(
            label,
            node_id,
            unique,
            shared,
            unique_label,
            shared_label,
            total_label,
            node_type,
            click_view=click_view,
        )
        if children:
            child_html = "".join(self._render_node(child) for child in children)
            return (
                f"<details class='ckpt-node'><summary>{row_html}</summary>"
                f"<div class='ckpt-children'>{child_html}</div></details>"
            )
        return f"<div class='ckpt-leaf'>{row_html}</div>"

    def _render_row(
        self,
        label,
        node_id,
        unique,
        shared,
        unique_label,
        shared_label,
        total_label,
        node_type,
        click_view=False,
    ):
        label_display = html.escape(label)
        label_attr = html.escape(label, quote=True)
        node_id_attr = html.escape(node_id, quote=True)
        type_attr = html.escape(node_type, quote=True)
        row_class = "ckpt-row clickable" if click_view else "ckpt-row"
        row_onclick = (
            "onclick='handlemodelRowClick(event, this);'" if click_view else ""
        )
        unique_badge = f"<span class='ckpt-badge ckpt-badge-unique'>{unique}</span>"
        shared_badge = f"<span class='ckpt-badge ckpt-badge-shared'>{shared}</span>"
        return (
            f"<div class='{row_class}' data-node-id=\"{node_id_attr}\" data-node-label=\"{label_attr}\" data-node-type=\"{type_attr}\" {row_onclick}>"
            f"<div class='ckpt-label'>{label_display}</div>"
            "<div class='ckpt-sizes'>"
            f"<div class='ckpt-size-unique' title='Unique size'>{unique_badge}</div>"
            f"<div class='ckpt-size-shared' title='Shared size'>{shared_badge}</div>"
            "</div>"
            "<div class='ckpt-actions'>"
            f"<button class='ckpt-action-btn ckpt-view-btn' data-node-id=\"{node_id_attr}\" data-node-label=\"{label_attr}\" data-node-type=\"{type_attr}\" data-unique-size=\"{unique_label}\" data-shared-size=\"{shared_label}\" data-total-size=\"{total_label}\" onclick='event.stopPropagation(); handlemodelAction(this, \"view\");'>View</button>"
            f"<button class='ckpt-action-btn ckpt-delete' data-node-id=\"{node_id_attr}\" data-node-label=\"{label_attr}\" data-node-type=\"{type_attr}\" data-unique-size=\"{unique_label}\" data-shared-size=\"{shared_label}\" data-total-size=\"{total_label}\" onclick='event.stopPropagation(); handlemodelAction(this, \"delete\");'>Delete</button>"
            "</div>"
            "</div>"
        )

    def _build_empty_view_html(self):
        return (
            "<div class='ckpt-view-sticky'><div class='ckpt-view'>"
            "<h4>Files</h4>"
            "<div class='ckpt-view-summary'>Select a branch to view its files.</div>"
            "</div></div>"
        )

    def _build_loading_html(self):
        return (
            "<div id='ckpt_tree_root' data-status='loading' class='ckpt-wrap'>"
            "<div class='ckpt-panel'><div class='ckpt-panel-inner'>"
            "<div class='ckpt-loading-message'>Please wait while the models hierarchy is being built...</div>"
            "</div></div></div>"
        )

    def _build_view_html(self, node_id):
        node = self._node_map.get(node_id)
        if node is None:
            return self._build_empty_view_html()

        files = sorted(node["files"])
        node_type = node.get("type", "")
        current_base = self._get_node_base_label(node_id, node_type)
        shared_set = {
            path
            for path in files
            if not self._file_usage_models.get(path, set()).issubset(node["models"])
        }

        current_base_label = None
        if current_base:
            current_base_label = f"{current_base[0]} {current_base[1]}"
        shared_info = {}
        for path in shared_set:
            shared_labels = self._shared_base_labels(path, exclude_key=current_base)
            if not shared_labels and current_base_label:
                shared_labels = [f"Shared only within {current_base_label}"]
            shared_info[path] = shared_labels
        file_tree = self._build_file_tree(files, shared_set, shared_info)
        file_tree_html = self._render_file_tree(file_tree)

        unique = self._format_gb(node["unique_size"])
        shared = self._format_gb(node["shared_size"])
        label_text = node["label"]
        path_label = node.get("path_label") or label_text
        header = html.escape(path_label)
        unique_badge = f"<span class='ckpt-badge ckpt-badge-unique'>{unique}</span>"
        shared_badge = f"<span class='ckpt-badge ckpt-badge-shared'>{shared}</span>"
        shared_count = len(shared_set)
        unique_count = max(0, len(files) - shared_count)
        summary = (
            f"<span class='ckpt-muted'>{unique_count} files</span> | unique {unique_badge} "
            f"<span class='ckpt-muted'>{shared_count} files</span> | shared {shared_badge}"
        )
        primary_paths = node.get("primary_paths", [])
        primary_html = ""
        if primary_paths:
            label = "model path" if len(primary_paths) == 1 else "model paths"
            lines = "<br>".join(html.escape(path) for path in primary_paths)
            primary_html = f"<div class='ckpt-primary'><strong>{label}:</strong><br>{lines}</div>"
        return (
            "<div class='ckpt-view-sticky'><div class='ckpt-view'>"
            f"<h4><strong>{header}</strong></h4>"
            f"{primary_html}"
            f"<div class='ckpt-view-summary'>{summary}</div>"
            f"{file_tree_html}</div></div>"
        )

    def _build_file_tree(self, files, shared_set, shared_info=None):
        if shared_info is None:
            shared_info = {}
        root = {
            "children": {},
            "files": [],
            "size": 0,
            "all_shared": False,
            "all_unique": False,
            "has_files": False,
            "shared_with": set(),
            "shared_count": 0,
            "is_root": True,
        }
        for path in files:
            info = self._file_info.get(path, {})
            display = self._display_path(path)
            parts = self._split_display_path(display)
            name = parts[-1] if parts else display
            shared_with = shared_info.get(path, [])
            entry = {
                "path": path,
                "display": display,
                "name": name,
                "size": info.get("size", 0),
                "exists": info.get("exists", False),
                "shared": path in shared_set,
                "shared_with": shared_with,
            }

            node = root
            for part in parts[:-1]:
                node = node["children"].setdefault(
                    part,
                    {
                        "children": {},
                        "files": [],
                        "size": 0,
                        "all_shared": False,
                        "all_unique": False,
                        "has_files": False,
                        "shared_with": set(),
                        "shared_count": 0,
                        "is_root": False,
                    },
                )
            node["files"].append(entry)

        self._compute_folder_sizes(root)
        return root

    def _compute_folder_sizes(self, node):
        total = sum(entry["size"] for entry in node["files"])
        has_files = len(node["files"]) > 0
        all_shared = all(entry["shared"] for entry in node["files"]) if node["files"] else True
        all_unique = all(not entry["shared"] for entry in node["files"]) if node["files"] else True
        shared_with = set()
        shared_count = 0
        for entry in node["files"]:
            if entry.get("shared"):
                shared_with.update(entry.get("shared_with", []))
        for child in node["children"].values():
            (
                child_total,
                child_has_files,
                child_all_shared,
                child_shared_with,
                child_shared_count,
                child_all_unique,
            ) = self._compute_folder_sizes(child)
            total += child_total
            has_files = has_files or child_has_files
            all_shared = all_shared and child_all_shared
            all_unique = all_unique and child_all_unique
            shared_with.update(child_shared_with)
            shared_count += child_shared_count
        node["size"] = total
        node["has_files"] = has_files
        node["all_shared"] = has_files and all_shared
        node["all_unique"] = has_files and all_unique
        node["shared_with"] = shared_with
        if node["all_shared"] and not node.get("is_root"):
            shared_count += 1
        node["shared_count"] = shared_count
        return total, has_files, node["all_shared"], shared_with, shared_count, node["all_unique"]

    def _render_file_tree(self, node):
        open_shared_path = node.get("shared_count", 0) == 1
        parts = []
        for name in sorted(node["children"].keys(), key=lambda n: n.lower()):
            parts.append(
                self._render_folder(
                    name,
                    node["children"][name],
                    level=0,
                    open_shared_path=open_shared_path,
                )
            )

        for entry in sorted(node["files"], key=lambda e: e["name"].lower()):
            parts.append(self._render_file_entry(entry))

        return "<div class='ckpt-file-tree'>" + "".join(parts) + "</div>"

    def _render_folder(self, name, node, level, open_shared_path=False):
        name, node = self._collapse_single_child_folder(name, node)
        folder_name = html.escape(name)
        size_label = self._format_size(node.get("size", 0))
        shared_tag = ""
        shared_tooltip = ""
        folder_class = "ckpt-folder"
        if node.get("all_shared"):
            shared_tag = "<span class='ckpt-shared-tag ckpt-folder-tag'>Shared</span>"
            folder_class += " shared"
            shared_with = sorted(node.get("shared_with", []), key=lambda value: value.lower())
            if shared_with:
                lines = "".join(
                    f"<div>{html.escape(label)}</div>" for label in shared_with
                )
                shared_tooltip = (
                    "<div class='ckpt-shared-tooltip'>"
                    "<div class='ckpt-shared-tooltip-title'>Shared with</div>"
                    f"{lines}</div>"
                )
        elif node.get("all_unique"):
            folder_class += " unique"
        open_attr = ""
        if level < 1 or (open_shared_path and node.get("shared_count", 0) == 1):
            open_attr = " open"
        children_html = []
        for child_name in sorted(node["children"].keys(), key=lambda n: n.lower()):
            children_html.append(
                self._render_folder(
                    child_name,
                    node["children"][child_name],
                    level + 1,
                    open_shared_path=open_shared_path,
                )
            )
        for entry in sorted(node["files"], key=lambda e: e["name"].lower()):
            children_html.append(self._render_file_entry(entry))
        return (
            f"<details class='{folder_class}'{open_attr}>"
            "<summary>"
            f"<span class='ckpt-folder-name'>{folder_name}</span>"
            f"{shared_tag}"
            f"<span class='ckpt-folder-size'>{size_label}</span>"
            f"{shared_tooltip}"
            "</summary>"
            "<div class='ckpt-folder-children'>"
            + "".join(children_html)
            + "</div></details>"
        )

    def _collapse_single_child_folder(self, name, node):
        parts = [name]
        current = node
        while True:
            if current.get("files"):
                break
            children = current.get("children", {})
            if len(children) != 1:
                break
            child_name, child_node = next(iter(children.items()))
            parts.append(child_name)
            current = child_node
        return os.sep.join(parts), current

    def _render_file_entry(self, entry):
        classes = ["ckpt-file-row"]
        if entry.get("shared"):
            classes.append("shared")
        else:
            classes.append("unique")
        if not entry.get("exists"):
            classes.append("missing")
        class_attr = " ".join(classes)
        name = html.escape(entry.get("name", ""))
        size_label = self._format_size(entry.get("size", 0))
        display = html.escape(entry.get("display", ""))
        shared_tag = "<span class='ckpt-shared-tag'>Shared</span>" if entry.get("shared") else ""
        shared_tooltip = ""
        if entry.get("shared"):
            shared_with = entry.get("shared_with", [])
            if shared_with:
                lines = "".join(
                    f"<div>{html.escape(label)}</div>" for label in shared_with
                )
                shared_tooltip = (
                    "<div class='ckpt-shared-tooltip'>"
                    "<div class='ckpt-shared-tooltip-title'>Shared with</div>"
                    f"{lines}</div>"
                )
        return (
            f"<div class='{class_attr}' title='{display}'>"
            f"<span class='ckpt-file-name'>{name}</span>"
            f"{shared_tag}"
            f"<span class='ckpt-file-size'>{size_label}</span>"
            f"{shared_tooltip}"
            "</div>"
        )

    def _split_display_path(self, display_path):
        drive, rest = os.path.splitdrive(display_path)
        if drive:
            drive = drive.rstrip("\\/")
        rest = rest.replace("\\", "/")
        parts = [part for part in rest.split("/") if part]
        if drive:
            return [drive] + parts
        if display_path.startswith("\\\\"):
            return ["UNC"] + parts
        return parts

    def _add_file(self, files_set, value, force_folder=None):
        path = self._resolve_path(value, force_folder=force_folder)
        if path and os.path.isfile(path):
            files_set.add(path)

    def _download_root_name(self):
        root = fl.get_download_location()
        if not root or os.path.isabs(root):
            return ""
        norm = os.path.normpath(root)
        if not norm or norm == ".":
            return ""
        return norm.split(os.sep)[0]

    def _resolve_repo_relative_path(self, rel_path):
        if not rel_path:
            return None
        rel_path = str(rel_path)
        if rel_path.startswith("http"):
            return None
        root_name = self._download_root_name()
        if not root_name:
            return None
        rel_norm = os.path.normpath(rel_path)
        parts = rel_norm.split(os.sep)
        if parts and parts[0].lower() == root_name.lower():
            return self._normalize_path(os.path.join(self._repo_root, rel_norm))
        return None

    def _resolve_path(self, value, force_folder=None):
        if not value:
            return None
        if not isinstance(value, str):
            value = str(value)
        if os.path.isabs(value):
            abs_path = self._normalize_path(value)
            if os.path.isfile(abs_path):
                return abs_path
            basename = os.path.basename(abs_path)
            return abs_path
        repo_rel = self._resolve_repo_relative_path(value)
        if repo_rel:
            return repo_rel

        try:
            local_path = self.get_local_model_filename(value, extra_paths=force_folder)
        except Exception:
            local_path = None
        if local_path:
            return self._normalize_path(local_path)

        filename = os.path.basename(value) if value.startswith("http") else value
        if force_folder:
            expected = fl.get_download_location(filename, force_path=force_folder)
        else:
            expected = fl.get_download_location(filename)
        return self._normalize_path(expected)

    def _resolve_download_relpath(self, rel_path):
        if rel_path and os.path.isabs(rel_path):
            return self._normalize_path(rel_path)
        repo_rel = self._resolve_repo_relative_path(rel_path)
        if repo_rel:
            return repo_rel
        local = fl.locate_file(rel_path, error_if_none=False)
        if local:
            return self._normalize_path(local)
        return self._normalize_path(fl.get_download_location(rel_path))

    def _resolve_folder_path(self, source_folder, target_folder):
        parts = []
        if target_folder:
            parts.append(target_folder)
        if source_folder:
            parts.append(source_folder)
        rel_path = os.path.join(*parts) if parts else ""
        if rel_path and os.path.isabs(rel_path):
            return self._normalize_path(rel_path)
        repo_rel = self._resolve_repo_relative_path(rel_path)
        if repo_rel:
            return repo_rel
        if not rel_path:
            return self._normalize_path(fl.get_download_location())
        folder = fl.locate_folder(rel_path, error_if_none=False)
        if folder:
            return self._normalize_path(folder)
        return self._normalize_path(fl.get_download_location(rel_path))

    def _list_folder_files(self, folder_path):
        files = set()
        if not folder_path or not os.path.isdir(folder_path):
            return files
        for root, _, filenames in os.walk(folder_path):
            for name in filenames:
                files.add(self._normalize_path(os.path.join(root, name)))
        return files

    def _combine_download_path(self, target_folder, source_folder, filename):
        parts = []
        if target_folder:
            parts.append(target_folder)
        if source_folder:
            parts.append(source_folder)
        parts.append(filename)
        return os.path.join(*parts)

    def _compute_list(self, filename):
        if not filename:
            return []
        return [os.path.basename(str(filename))]

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _safe_get_lora_dir(self, model_type):
        try:
            return self.get_lora_dir(model_type)
        except Exception:
            return None

    def _normalize_path(self, path):
        return os.path.normcase(os.path.abspath(path))

    def _format_gb(self, size_bytes):
        size_gb = size_bytes / (1024 ** 3)
        return f"{size_gb:.2f} GB"

    def _format_size(self, size_bytes):
        if size_bytes >= 1024 ** 3:
            return f"{size_bytes / (1024 ** 3):.2f} GB"
        if size_bytes >= 1024 ** 2:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        if size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes} B"

    def _display_path(self, path):
        try:
            rel = os.path.relpath(path, self._repo_root)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
        return path

    def _get_model_roots(self):
        roots = []
        for root in getattr(fl, "_models_paths", []) or []:
            if not root:
                continue
            if os.path.isabs(root):
                abs_root = root
            else:
                abs_root = os.path.join(self._repo_root, root)
            abs_root = self._normalize_path(abs_root)
            if abs_root not in roots:
                roots.append(abs_root)
        return roots

    def _build_basename_index(self, basenames, min_size=1):
        basenames = {name.lower() for name in basenames if name}
        if not basenames:
            return {}
        found = {}
        remaining = set(basenames)
        for root in self._get_model_roots():
            if not remaining:
                break
            if not os.path.isdir(root):
                continue
            for name in list(remaining):
                candidate = os.path.join(root, name)
                if os.path.isfile(candidate):
                    try:
                        if os.path.getsize(candidate) < min_size:
                            continue
                    except OSError:
                        continue
                    found[name] = self._normalize_path(candidate)
                    remaining.discard(name)
            if not remaining:
                break
            if os.path.normcase(root) == os.path.normcase(self._repo_root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    key = filename.lower()
                    if key in remaining:
                        full_path = os.path.join(dirpath, filename)
                        try:
                            if os.path.getsize(full_path) < min_size:
                                continue
                        except OSError:
                            continue
                        found[key] = self._normalize_path(full_path)
                        remaining.discard(key)
                        if not remaining:
                            break
                if not remaining:
                    break
        return found

    def _find_in_model_roots(self, basename, min_size=1):
        if not basename:
            return None
        key = basename.lower()
        if key in self._basename_index:
            return self._basename_index.get(key)
        found = self._build_basename_index({basename}, min_size=min_size)
        if found:
            self._basename_index.update(found)
            return self._basename_index.get(key)
        self._basename_index[key] = None
        return None

    def _repair_missing_paths(self):
        missing = []
        for path in self._file_usage_models.keys():
            try:
                size = os.path.getsize(path) if os.path.isfile(path) else -1
            except OSError:
                size = -1
            if size <= 0:
                missing.append(path)
        if not missing:
            return
        basenames = {os.path.basename(path) for path in missing}
        found = self._build_basename_index(basenames, min_size=1)
        if not found:
            return
        for old_path in list(missing):
            key = os.path.basename(old_path).lower()
            new_path = found.get(key)
            if not new_path or new_path == old_path:
                continue
            self._swap_path(old_path, new_path)

    def _swap_path(self, old_path, new_path):
        usage_ids = self._file_usage.pop(old_path, set())
        if usage_ids:
            self._file_usage[new_path].update(usage_ids)
            for usage_id in usage_ids:
                files = self._usage_files.get(usage_id)
                if files is not None:
                    files.discard(old_path)
                    files.add(new_path)
                model_type = self._variant_to_model.get(usage_id)
                if model_type:
                    for variant in self._model_variants.get(model_type, []):
                        if variant.get("id") != usage_id:
                            continue
                        files_set = variant.get("files")
                        if isinstance(files_set, set):
                            files_set.discard(old_path)
                            files_set.add(new_path)
                        primary_paths = variant.get("primary_paths", [])
                        variant["primary_paths"] = [
                            new_path if path == old_path else path for path in primary_paths
                        ]
                        break

        model_types = self._file_usage_models.pop(old_path, set())
        if model_types:
            self._file_usage_models[new_path].update(model_types)
            for model_type in model_types:
                files = self._model_files.get(model_type)
                if files is not None:
                    files.discard(old_path)
                    files.add(new_path)
                primary_paths = self._model_primary.get(model_type)
                if primary_paths:
                    self._model_primary[model_type] = [
                        new_path if path == old_path else path for path in primary_paths
                    ]
