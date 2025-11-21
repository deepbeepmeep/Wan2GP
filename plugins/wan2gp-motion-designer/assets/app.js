// Copyright WanGP. Subject to the WanGP license.
(function () {
    const EVENT_TYPE = "WAN2GP_MOTION_DESIGNER";
    const CONTROL_MESSAGE_TYPE = "WAN2GP_MOTION_DESIGNER_CONTROL";
    const RENDER_MODES = {
        CUT_DRAG: "cut_drag",
        CLASSIC: "classic",
    };
    const MIME_CANDIDATES = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
    ];
    const COLOR_POOL = ["#4cc2ff", "#ff9f43", "#8d78ff", "#ff5e8a", "#32d5a4", "#f9d65c"];
    const POLYGON_EDGE_COLOR = "#ff4d57";
    const TRAJECTORY_EDGE_COLOR = "#4d9bff";
    const DEFAULT_CLASSIC_OUTLINE_WIDTH = 1;

    const state = {
        baseImage: null,
        baseCanvas: document.createElement("canvas"),
        backgroundCanvas: document.createElement("canvas"),
        altBackgroundImage: null,
        altBackgroundDataUrl: null,
        resolution: { width: 832, height: 480 },
        fitMode: "cover",
        scene: {
            fps: 16,
            totalFrames: 81,
        },
        layers: [],
        activeLayerId: null,
        showPatchedBackground: true,
        baseCanvasDataUrl: null,
        previewMode: false,
        classicOutlineWidth: DEFAULT_CLASSIC_OUTLINE_WIDTH,
        backgroundDataUrl: null,
        dragContext: null,
        renderMode: RENDER_MODES.CUT_DRAG,
        modeToggleVisible: true,
        animation: {
            playhead: 0,
            playing: false,
            loop: true,
            lastTime: 0,
        },
        transfer: {
            pending: false,
            mask: null,
            guide: null,
            backgroundImage: null,
        },
        export: {
            running: false,
            mode: null,
            variant: "mask",
            renderMode: RENDER_MODES.CUT_DRAG,
            canvas: null,
            ctx: null,
            recorder: null,
            stream: null,
            chunks: [],
            frame: 0,
            totalFrames: 0,
            mimeType: null,
            cancelled: false,
        },
        currentWorkflowStage: "scene",
        userPanelOverride: null,
        currentExpandedPanel: null,
        shapeMode: "rectangle",
        theme: "light",
    };

    const dom = {};

    document.addEventListener("DOMContentLoaded", init);

    function init() {
        cacheDom();
        applyTheme(state.theme);
        syncSceneInputs();
        bindEvents();
        initCollapsiblePanels();
        matchCanvasResolution();
        resetLayers();
        updatePreviewModeUI();
        updateWorkflowPanels();
        updateSpeedControlVisibility();
        updateSpeedRatioLabel();
        updateCanvasPlaceholder();
        updateModeToggleUI();
        applyModeToggleVisibility();
        updateClassicOutlineControls();
        initExternalBridge();
        setupHeightObserver();
        requestAnimationFrame(render);
        updateBackgroundToggleUI();
    }

    function setModalOpen(open) {
        const locked = Boolean(open);
        document.documentElement.classList.toggle("modal-open", locked);
        document.body.classList.toggle("modal-open", locked);
        try {
            window.parent?.postMessage({ type: "WAN2GP_MOTION_DESIGNER_MODAL_LOCK", open: locked }, "*");
        } catch (err) {
            console.warn("[MotionDesigner] Unable to notify parent about modal state", err);
        }
    }

    function cacheDom() {
        dom.canvas = document.getElementById("editorCanvas");
        dom.ctx = dom.canvas.getContext("2d");
        dom.badge = document.getElementById("canvasGuide");
        dom.statusBadge = document.getElementById("statusBadge");
        dom.sourceChooser = document.getElementById("sourceChooser");
        dom.targetWidthInput = document.getElementById("targetWidthInput");
        dom.targetHeightInput = document.getElementById("targetHeightInput");
        dom.fitModeSelect = document.getElementById("fitModeSelect");
        dom.applyResolutionBtn = document.getElementById("applyResolutionBtn");
        dom.sceneFpsInput = document.getElementById("sceneFpsInput");
        dom.sceneFrameCountInput = document.getElementById("sceneFrameCountInput");
        dom.scaleStartInput = document.getElementById("scaleStartInput");
        dom.scaleEndInput = document.getElementById("scaleEndInput");
        dom.rotationStartInput = document.getElementById("rotationStartInput");
        dom.rotationEndInput = document.getElementById("rotationEndInput");
        dom.speedModeSelect = document.getElementById("speedModeSelect");
        dom.speedRatioInput = document.getElementById("speedRatioInput");
        dom.speedRatioLabel = document.getElementById("speedRatioLabel");
        dom.speedRatioRow = document.getElementById("speedRatioRow");
        dom.speedRatioValue = document.getElementById("speedRatioValue");
        dom.hideOutsideRangeToggle = document.getElementById("hideOutsideRangeToggle");
        dom.tensionInput = document.getElementById("tensionInput");
        dom.tensionValueLabel = document.getElementById("tensionValue");
        dom.startFrameInput = document.getElementById("startFrameInput");
        dom.endFrameInput = document.getElementById("endFrameInput");
        dom.timelineSlider = document.getElementById("timelineSlider");
        dom.previewModeBtn = document.getElementById("previewModeBtn");
        dom.playPauseBtn = document.getElementById("playPauseBtn");
        dom.loopToggle = document.getElementById("loopToggle");
        dom.classicOutlineControl = document.getElementById("classicOutlineControl");
        dom.classicOutlineSlider = document.getElementById("classicOutlineSlider");
        dom.classicOutlineValue = document.getElementById("classicOutlineValue");
        dom.modeSwitcher = document.getElementById("modeSwitcher");
        dom.modeCutDragBtn = document.getElementById("modeCutDragBtn");
        dom.modeClassicBtn = document.getElementById("modeClassicBtn");
        dom.canvasFooter = document.getElementById("canvasFooter");
        dom.downloadMaskBtn = document.getElementById("downloadMaskBtn");
        dom.sendToWangpBtn = document.getElementById("sendToWangpBtn");
        dom.backgroundToggleBtn = document.getElementById("backgroundToggleBtn");
        dom.backgroundFileInput = document.getElementById("backgroundFileInput");
        dom.saveDefinitionBtn = document.getElementById("saveDefinitionBtn");
        dom.loadDefinitionBtn = document.getElementById("loadDefinitionBtn");
        dom.definitionFileInput = document.getElementById("definitionFileInput");
        dom.downloadBackgroundBtn = document.getElementById("downloadBackgroundBtn");
        dom.exportOverlay = document.getElementById("exportOverlay");
        dom.exportProgressBar = document.getElementById("exportProgressBar");
        dom.exportLabel = document.getElementById("exportLabel");
        dom.cancelExportBtn = document.getElementById("cancelExportBtn");
        dom.activeObjectLabel = document.getElementById("activeObjectLabel");
        dom.contextDeleteBtn = document.getElementById("contextDeleteBtn");
        dom.contextResetBtn = document.getElementById("contextResetBtn");
        dom.contextUndoBtn = document.getElementById("contextUndoBtn");
        dom.contextButtonGroup = document.getElementById("contextButtonGroup");
        dom.shapeModeSelect = document.getElementById("shapeModeSelect");
        dom.objectPanelBody = document.querySelector('.panel[data-panel="object"] .panel-body');
        dom.canvasPlaceholder = document.getElementById("canvasPlaceholder");
        dom.canvasUploadBtn = document.getElementById("canvasUploadBtn");
        dom.themeToggleBtn = document.getElementById("themeToggleBtn");
        dom.unloadSceneBtn = document.getElementById("unloadSceneBtn");
        dom.unloadOverlay = document.getElementById("unloadOverlay");
        dom.confirmUnloadBtn = document.getElementById("confirmUnloadBtn");
        dom.cancelUnloadBtn = document.getElementById("cancelUnloadBtn");
    }

    function applyTheme(theme) {
        state.theme = theme === "dark" ? "dark" : "light";
        document.body.classList.toggle("theme-dark", state.theme === "dark");
        if (dom.themeToggleBtn) {
            dom.themeToggleBtn.textContent = state.theme === "dark" ? "Light Mode" : "Dark Mode";
        }
    }

    function toggleTheme() {
        applyTheme(state.theme === "dark" ? "light" : "dark");
    }

    function bindEvents() {
        dom.sourceChooser.addEventListener("change", handleSourceFile);
        dom.applyResolutionBtn.addEventListener("click", () => {
            if (state.baseImage) {
                applyResolution();
            }
        });
        if (dom.classicOutlineSlider) {
            dom.classicOutlineSlider.addEventListener("input", (evt) => {
                handleClassicOutlineChange(evt.target.value);
            });
        }
        dom.sceneFpsInput.addEventListener("input", handleSceneFpsChange);
        dom.sceneFrameCountInput.addEventListener("change", handleSceneFrameCountChange);
        dom.sceneFrameCountInput.addEventListener("blur", handleSceneFrameCountChange);
        dom.scaleStartInput.addEventListener("input", () => handleLayerAnimInput("scaleStart", parseFloat(dom.scaleStartInput.value)));
        dom.scaleEndInput.addEventListener("input", () => handleLayerAnimInput("scaleEnd", parseFloat(dom.scaleEndInput.value)));
        dom.rotationStartInput.addEventListener("input", () => handleLayerAnimInput("rotationStart", parseFloat(dom.rotationStartInput.value)));
        dom.rotationEndInput.addEventListener("input", () => handleLayerAnimInput("rotationEnd", parseFloat(dom.rotationEndInput.value)));
        dom.speedModeSelect.addEventListener("change", () => {
            handleLayerAnimInput("speedMode", dom.speedModeSelect.value);
            updateSpeedControlVisibility();
        });
        dom.speedRatioInput.addEventListener("input", () => {
            handleLayerAnimInput("speedRatio", parseFloat(dom.speedRatioInput.value));
            updateSpeedRatioLabel();
        });
        dom.startFrameInput.addEventListener("input", () => handleLayerFrameInput("startFrame", parseInt(dom.startFrameInput.value, 10)));
        dom.endFrameInput.addEventListener("input", () => handleLayerFrameInput("endFrame", parseInt(dom.endFrameInput.value, 10)));
        dom.hideOutsideRangeToggle.addEventListener("change", (evt) => handleHideOutsideRangeToggle(evt.target.checked));
        dom.tensionInput.addEventListener("input", () => handleLayerAnimInput("tension", parseFloat(dom.tensionInput.value)));
        dom.timelineSlider.addEventListener("input", handleTimelineScrub);
        dom.previewModeBtn.addEventListener("click", togglePreviewMode);
        dom.playPauseBtn.addEventListener("click", togglePlayback);
        dom.loopToggle.addEventListener("change", (evt) => {
            state.animation.loop = evt.target.checked;
        });
        dom.downloadMaskBtn.addEventListener("click", () => startExport("download"));
        dom.sendToWangpBtn.addEventListener("click", handleSendToWangp);
        if (dom.backgroundToggleBtn && dom.backgroundFileInput) {
            dom.backgroundToggleBtn.addEventListener("click", handleBackgroundToggle);
            dom.backgroundFileInput.addEventListener("change", handleBackgroundFileSelected);
        }
        if (dom.saveDefinitionBtn) {
            dom.saveDefinitionBtn.addEventListener("click", handleSaveDefinition);
        }
        if (dom.loadDefinitionBtn && dom.definitionFileInput) {
            dom.loadDefinitionBtn.addEventListener("click", (evt) => {
                evt.preventDefault();
                dom.definitionFileInput.value = "";
                dom.definitionFileInput.click();
            });
            dom.definitionFileInput.addEventListener("change", handleDefinitionFileSelected);
        }
        dom.downloadBackgroundBtn.addEventListener("click", downloadBackgroundImage);
        dom.cancelExportBtn.addEventListener("click", cancelExport);
        dom.contextDeleteBtn.addEventListener("click", handleContextDelete);
        dom.contextResetBtn.addEventListener("click", handleContextReset);
        dom.contextUndoBtn.addEventListener("click", handleContextUndo);
        if (dom.unloadSceneBtn) {
            dom.unloadSceneBtn.addEventListener("click", promptUnloadScene);
        }
        if (dom.shapeModeSelect) {
            dom.shapeModeSelect.addEventListener("change", (evt) => {
                handleShapeModeSelectChange(evt.target.value || "polygon");
            });
            dom.shapeModeSelect.value = state.shapeMode;
        }
        if (dom.canvasUploadBtn && dom.sourceChooser) {
            dom.canvasUploadBtn.addEventListener("click", (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                dom.sourceChooser.click();
            });
        }
        if (dom.canvasPlaceholder && dom.sourceChooser) {
            dom.canvasPlaceholder.addEventListener("click", (evt) => {
                if (evt.target === dom.canvasUploadBtn) {
                    return;
                }
                dom.sourceChooser.click();
            });
        }
        if (dom.confirmUnloadBtn) {
            dom.confirmUnloadBtn.addEventListener("click", () => {
                hideUnloadPrompt();
                handleUnloadScene();
            });
        }
        if (dom.cancelUnloadBtn) {
            dom.cancelUnloadBtn.addEventListener("click", hideUnloadPrompt);
        }
        if (dom.themeToggleBtn) {
            dom.themeToggleBtn.addEventListener("click", (evt) => {
                evt.preventDefault();
                toggleTheme();
            });
        }
        if (dom.modeCutDragBtn) {
            dom.modeCutDragBtn.addEventListener("click", () => setRenderMode(RENDER_MODES.CUT_DRAG));
        }
        if (dom.modeClassicBtn) {
            dom.modeClassicBtn.addEventListener("click", () => setRenderMode(RENDER_MODES.CLASSIC));
        }

        dom.canvas.addEventListener("pointerdown", onPointerDown);
        dom.canvas.addEventListener("pointermove", onPointerMove);
        dom.canvas.addEventListener("pointerup", onPointerUp);
        dom.canvas.addEventListener("pointerleave", onPointerUp);
        dom.canvas.addEventListener("dblclick", onCanvasDoubleClick);
        dom.canvas.addEventListener("contextmenu", onCanvasContextMenu);
    }

    function initCollapsiblePanels() {
        dom.panelMap = {};
        dom.collapsiblePanels = Array.from(document.querySelectorAll(".panel.collapsible"));
        dom.collapsiblePanels.forEach((panel) => {
            const id = panel.dataset.panel;
            if (!id) {
                return;
            }
            const toggle = panel.querySelector(".panel-toggle");
            dom.panelMap[id] = { element: panel, toggle };
            if (panel.dataset.fixed === "true") {
                panel.classList.add("expanded");
                if (toggle) {
                    toggle.setAttribute("aria-disabled", "true");
                }
                return;
            }
            if (toggle) {
                toggle.addEventListener("click", () => {
                    expandPanel(id, true);
                });
            }
        });
        if (!state.currentExpandedPanel && dom.panelMap.scene) {
            expandPanel("scene");
        }
    }

    function setRenderMode(mode) {
        const normalized = mode === RENDER_MODES.CLASSIC ? RENDER_MODES.CLASSIC : RENDER_MODES.CUT_DRAG;
        if (state.renderMode === normalized) {
            updateModeToggleUI();
            updateClassicOutlineControls();
            return;
        }
        state.renderMode = normalized;
        updateModeToggleUI();
        updateClassicOutlineControls();
        updateBackgroundToggleUI();
    }

    function updateModeToggleUI() {
        if (!dom.modeCutDragBtn || !dom.modeClassicBtn) {
            return;
        }
        const isClassic = state.renderMode === RENDER_MODES.CLASSIC;
        dom.modeCutDragBtn.classList.toggle("active", !isClassic);
        dom.modeClassicBtn.classList.toggle("active", isClassic);
        dom.modeCutDragBtn.setAttribute("aria-pressed", (!isClassic).toString());
        dom.modeClassicBtn.setAttribute("aria-pressed", isClassic.toString());
    }

    function setModeToggleVisibility(visible) {
        state.modeToggleVisible = Boolean(visible);
        applyModeToggleVisibility();
    }

    function applyModeToggleVisibility() {
        if (!dom.modeSwitcher) {
            return;
        }
        dom.modeSwitcher.classList.toggle("is-hidden", !state.modeToggleVisible);
        updateBackgroundToggleUI();
    }

    function initExternalBridge() {
        if (typeof window === "undefined") {
            return;
        }
        window.addEventListener("message", handleExternalControlMessage);
        window.Wan2gpCutDrag = {
            setMode: (mode) => setRenderMode(mode),
            setModeToggleVisibility: (visible) => setModeToggleVisibility(visible),
        };
    }

    function handleExternalControlMessage(event) {
        if (!event || typeof event.data !== "object" || event.data === null) {
            return;
        }
        const payload = event.data;
        if (payload.type !== CONTROL_MESSAGE_TYPE) {
            return;
        }
        switch (payload.action) {
            case "setMode":
                setRenderMode(payload.value);
                break;
            case "setModeToggleVisibility":
                setModeToggleVisibility(payload.value !== false);
                break;
            default:
                break;
        }
    }

    function setStatus(message, variant = "muted") {
        if (!dom.statusBadge) {
            return;
        }
        dom.statusBadge.textContent = message;
        dom.statusBadge.classList.remove("muted", "success", "warn", "error");
        dom.statusBadge.classList.add(
            variant === "success" ? "success" : variant === "warn" ? "warn" : variant === "error" ? "error" : "muted",
        );
    }

    function notify(message, variant = "muted") {
        setStatus(message, variant);
        console[variant === "error" ? "error" : variant === "warn" ? "warn" : "log"]("[MotionDesigner]", message);
    }
    function matchCanvasResolution() {
        state.baseCanvas.width = state.resolution.width;
        state.baseCanvas.height = state.resolution.height;
        state.backgroundCanvas.width = state.resolution.width;
        state.backgroundCanvas.height = state.resolution.height;
        dom.canvas.width = state.resolution.width;
        dom.canvas.height = state.resolution.height;
    }

    function resetLayers() {
        state.layers = [];
        state.activeLayerId = null;
        if (state.baseImage) {
            addLayer(state.shapeMode);
        }
        state.backgroundDataUrl = null;
        if (dom.timelineSlider) {
            dom.timelineSlider.value = "0";
        }
        state.animation.playhead = 0;
        state.animation.playing = false;
        if (dom.playPauseBtn) {
            dom.playPauseBtn.textContent = "Play";
        }
        updateBadge();
        updateActionAvailability();
        updateCanvasPlaceholder();
        updateLayerAnimationInputs();
    }

    function createLayer(name, colorIndex, shapeOverride) {
        return {
            id: typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `layer_${Date.now()}_${Math.random()}`,
            name,
            color: COLOR_POOL[colorIndex % COLOR_POOL.length],
            polygon: [],
            polygonClosed: false,
            polygonPreviewPoint: null,
            selectedPolygonIndex: -1,
            path: [],
            pathLocked: false,
            selectedPathIndex: -1,
            localPolygon: [],
            anchor: { x: 0, y: 0 },
            objectCut: null,
            pathMeta: null,
            renderPath: [],
            renderSegmentMap: [],
            shapeType: shapeOverride || state.shapeMode || "polygon",
            shapeDraft: null,
            tempPolygon: null,
            shapeMeta: null,
            startFrame: 0,
            endFrame: state.scene.totalFrames,
            scaleStart: 1,
            scaleEnd: 1,
            rotationStart: 0,
            rotationEnd: 0,
            speedMode: "none",
            speedRatio: 1,
            tension: 0,
            hideOutsideRange: true,
        };
    }

    function handleShapeModeSelectChange(mode) {
        const normalized = mode || "polygon";
        state.shapeMode = normalized;
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        const stage = getLayerStage(layer);
        const hasPolygonPoints = Array.isArray(layer.polygon) && layer.polygon.length > 0;
        if (stage === "polygon" && !layer.polygonClosed && !hasPolygonPoints) {
            layer.shapeType = normalized;
            layer.shapeDraft = null;
            layer.tempPolygon = null;
            layer.shapeMeta = null;
            layer.polygonPreviewPoint = null;
        }
    }

    function addLayer(shapeOverride) {
        const layer = createLayer(`Object ${state.layers.length + 1}`, state.layers.length, shapeOverride);
        state.layers.push(layer);
        state.activeLayerId = layer.id;
        updateLayerPathCache(layer);
        refreshLayerSelect();
        updateBindings();
        updateBadge();
        updateActionAvailability();
        return layer;
    }

    function removeLayer() {
        if (!state.activeLayerId) {
            return;
        }
        state.layers = state.layers.filter((layer) => layer.id !== state.activeLayerId);
        state.activeLayerId = state.layers[0]?.id || null;
        refreshLayerSelect();
        updateBindings();
        recomputeBackgroundFill();
        updateBadge();
        updateActionAvailability();
    }

    function refreshLayerSelect() {
        updateLayerAnimationInputs();
    }

    function updateLayerAnimationInputs() {
        if (!dom.activeObjectLabel) {
            return;
        }
        const layer = getActiveLayer();
        if (dom.objectPanelBody) {
            dom.objectPanelBody.classList.toggle("disabled", !layer);
        }
        const layerIndex = layer ? state.layers.findIndex((item) => item.id === layer.id) : -1;
        dom.activeObjectLabel.textContent = layerIndex >= 0 ? String(layerIndex + 1) : "None";
        if (dom.shapeModeSelect) {
            dom.shapeModeSelect.value = layer ? layer.shapeType || "polygon" : state.shapeMode;
        }
        const inputs = [
            dom.scaleStartInput,
            dom.scaleEndInput,
            dom.rotationStartInput,
            dom.rotationEndInput,
            dom.startFrameInput,
            dom.endFrameInput,
            dom.speedModeSelect,
            dom.speedRatioInput,
            dom.tensionInput,
        ];
        inputs.forEach((input) => {
            if (input) {
                input.disabled = !layer;
                if (input === dom.startFrameInput || input === dom.endFrameInput) {
                    input.min = 0;
                    input.max = state.scene.totalFrames;
                }
            }
        });
        if (dom.hideOutsideRangeToggle) {
            dom.hideOutsideRangeToggle.disabled = !layer;
        }
        if (!layer) {
            if (dom.scaleStartInput) dom.scaleStartInput.value = 1;
            if (dom.scaleEndInput) dom.scaleEndInput.value = 1;
            if (dom.rotationStartInput) dom.rotationStartInput.value = 0;
            if (dom.rotationEndInput) dom.rotationEndInput.value = 0;
            if (dom.startFrameInput) dom.startFrameInput.value = 0;
            if (dom.endFrameInput) dom.endFrameInput.value = state.scene.totalFrames;
            if (dom.speedModeSelect) dom.speedModeSelect.value = "none";
            if (dom.speedRatioInput) dom.speedRatioInput.value = 1;
            if (dom.tensionInput) dom.tensionInput.value = 0;
            if (dom.tensionValueLabel) dom.tensionValueLabel.textContent = "0%";
            if (dom.hideOutsideRangeToggle) dom.hideOutsideRangeToggle.checked = true;
            updateSpeedControlVisibility(null);
            updateSpeedRatioLabel();
            return;
        }
        if (dom.scaleStartInput) dom.scaleStartInput.value = layer.scaleStart ?? 1;
        if (dom.scaleEndInput) dom.scaleEndInput.value = layer.scaleEnd ?? 1;
        if (dom.rotationStartInput) dom.rotationStartInput.value = layer.rotationStart ?? 0;
        if (dom.rotationEndInput) dom.rotationEndInput.value = layer.rotationEnd ?? 0;
        if (dom.startFrameInput) dom.startFrameInput.value = layer.startFrame ?? 0;
        if (dom.endFrameInput) dom.endFrameInput.value = layer.endFrame ?? state.scene.totalFrames;
        if (dom.speedModeSelect) dom.speedModeSelect.value = layer.speedMode || "none";
        if (dom.speedRatioInput) dom.speedRatioInput.value = layer.speedRatio ?? 1;
        if (dom.tensionInput) dom.tensionInput.value = layer.tension ?? 0;
        if (dom.tensionValueLabel) dom.tensionValueLabel.textContent = `${Math.round((layer.tension ?? 0) * 100)}%`;
        if (dom.hideOutsideRangeToggle) dom.hideOutsideRangeToggle.checked = layer.hideOutsideRange !== false;
        updateSpeedControlVisibility(layer);
        updateSpeedRatioLabel();
    }

    function setActiveLayer(layerId) {
        if (!layerId) {
            state.activeLayerId = null;
            updateBindings();
            updateBadge();
            updateActionAvailability();
            return;
        }
        const layer = state.layers.find((item) => item.id === layerId);
        if (!layer) {
            return;
        }
        state.activeLayerId = layer.id;
        ensureTrajectoryEditable(layer);
        updateBindings();
        updateBadge();
        updateActionAvailability();
    }

    function updateBindings() {
        const layer = getActiveLayer();
        updateLayerAnimationInputs();
    }

    function getActiveLayer() {
        return state.layers.find((layer) => layer.id === state.activeLayerId) || null;
    }

    function getLayerById(layerId) {
        if (!layerId) {
            return null;
        }
        return state.layers.find((layer) => layer.id === layerId) || null;
    }

    function isLayerEmpty(layer) {
        if (!layer) {
            return false;
        }
        const hasPolygon = layer.polygonClosed || layer.polygon.length > 0;
        const hasPath = Array.isArray(layer.path) && layer.path.length > 0;
        const hasCut = Boolean(layer.objectCut);
        return !hasPolygon && !hasPath && !hasCut;
    }

    function handleSourceFile(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) {
            return;
        }
        state.baseImage = null;
        updateCanvasPlaceholder();
        resetLayers();
        setStatus("Loading scene...", "warn");
        const reader = new FileReader();
        reader.onload = async () => {
            try {
                await loadBaseImage(reader.result);
                applyResolution();
                setStatus("Scene ready. Draw the polygon.", "success");
            } catch (err) {
                console.error("Failed to load scene", err);
                setStatus("Unable to load the selected file.", "error");
            }
        };
        reader.onerror = () => setStatus("Unable to read the selected file.", "error");
        reader.readAsDataURL(file);
    }

    async function loadBaseImage(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                state.baseImage = img;
                updateCanvasPlaceholder();
                const width = clamp(Math.round(img.width) || 832, 128, 4096);
                const height = clamp(Math.round(img.height) || 480, 128, 4096);
                if (dom.targetWidthInput) {
                    dom.targetWidthInput.value = width;
                }
                if (dom.targetHeightInput) {
                    dom.targetHeightInput.value = height;
                }
                if (state.layers.length === 0) {
                    addLayer(state.shapeMode);
                }
                resolve();
            };
            img.onerror = reject;
            img.src = dataUrl;
        });
    }

    function applyResolution() {
        const width = clamp(Number(dom.targetWidthInput.value) || 832, 128, 4096);
        const height = clamp(Number(dom.targetHeightInput.value) || 480, 128, 4096);
        dom.targetWidthInput.value = width;
        dom.targetHeightInput.value = height;
        state.resolution = { width, height };
        state.fitMode = dom.fitModeSelect.value;
        matchCanvasResolution();
        if (state.baseImage) {
            const ctx = state.baseCanvas.getContext("2d");
            ctx.clearRect(0, 0, width, height);
            const ratio =
                state.fitMode === "cover"
                    ? Math.max(width / state.baseImage.width, height / state.baseImage.height)
                    : Math.min(width / state.baseImage.width, height / state.baseImage.height);
            const newWidth = state.baseImage.width * ratio;
            const newHeight = state.baseImage.height * ratio;
            const offsetX = (width - newWidth) / 2;
            const offsetY = (height - newHeight) / 2;
            ctx.drawImage(state.baseImage, offsetX, offsetY, newWidth, newHeight);
            state.baseCanvasDataUrl = state.baseCanvas.toDataURL("image/png");
        } else {
            state.baseCanvasDataUrl = null;
        }
        state.layers.forEach((layer) => resetLayerGeometry(layer));
        recomputeBackgroundFill();
        updateBadge();
        updateActionAvailability();
    }

    function resetLayerGeometry(layer) {
        layer.polygon = [];
        layer.polygonClosed = false;
        layer.polygonPreviewPoint = null;
        layer.selectedPolygonIndex = -1;
        layer.path = [];
        layer.pathLocked = false;
        layer.selectedPathIndex = -1;
        layer.localPolygon = [];
        layer.anchor = { x: 0, y: 0 };
        layer.objectCut = null;
        layer.pathMeta = null;
        layer.renderPath = [];
        layer.renderSegmentMap = [];
        layer.shapeType = layer.shapeType || state.shapeMode || "polygon";
        layer.shapeDraft = null;
        layer.tempPolygon = null;
        layer.shapeMeta = null;
        layer.startFrame = 0;
        layer.endFrame = state.scene.totalFrames;
        layer.scaleStart = 1;
        layer.scaleEnd = 1;
        layer.rotationStart = 0;
        layer.rotationEnd = 0;
        layer.speedMode = layer.speedMode || "none";
        layer.speedRatio = layer.speedRatio ?? 1;
        layer.tension = 0;
        layer.hideOutsideRange = true;
        updateLayerPathCache(layer);
    }
    function finishPolygon() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        if (layer.polygonClosed) {
            computeLayerAssets(layer);
            return;
        }
        if (layer.polygon.length < 3) {
            setStatus("Add at least three points for the polygon.", "warn");
            return;
        }
        layer.polygonClosed = true;
        computeLayerAssets(layer);
        setStatus("Polygon closed. Draw the trajectory.", "success");
        enterTrajectoryCreationMode(layer, true);
        updateBadge();
        updateActionAvailability();
    }

    function undoPolygonPoint() {
        const layer = getActiveLayer();
        if (!layer || layer.polygon.length === 0) {
            return;
        }
        layer.polygon.pop();
        layer.polygonClosed = false;
        layer.objectCut = null;
        recomputeBackgroundFill();
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
    }

    function resetPolygon() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        layer.polygon = [];
        layer.polygonClosed = false;
        layer.polygonPreviewPoint = null;
        layer.shapeDraft = null;
        layer.tempPolygon = null;
        layer.shapeMeta = null;
        layer.objectCut = null;
        recomputeBackgroundFill();
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
    }

    function computeLayerAssets(layer) {
        if (!state.baseImage || !layer.polygonClosed || layer.polygon.length < 3) {
            return;
        }
        const bounds = polygonBounds(layer.polygon);
        const width = Math.max(1, Math.round(bounds.width));
        const height = Math.max(1, Math.round(bounds.height));
        const cutCanvas = document.createElement("canvas");
        cutCanvas.width = width;
        cutCanvas.height = height;
        const ctx = cutCanvas.getContext("2d");
        ctx.save();
        ctx.translate(-bounds.minX, -bounds.minY);
        drawPolygonPath(ctx, layer.polygon);
        ctx.clip();
        ctx.drawImage(state.baseCanvas, 0, 0);
        ctx.restore();
        layer.localPolygon = layer.polygon.map((pt) => ({ x: pt.x - bounds.minX, y: pt.y - bounds.minY }));
        const centroid = polygonCentroid(layer.polygon);
        layer.anchor = { x: centroid.x - bounds.minX, y: centroid.y - bounds.minY };
        layer.objectCut = { canvas: cutCanvas };
        recomputeBackgroundFill();
        updateLayerPathCache(layer);
    }

    function recomputeBackgroundFill() {
        const ctx = state.backgroundCanvas.getContext("2d");
        ctx.clearRect(0, 0, state.backgroundCanvas.width, state.backgroundCanvas.height);
        ctx.drawImage(state.baseCanvas, 0, 0);
        const closedLayers = state.layers.filter((layer) => layer.polygonClosed && layer.polygon.length >= 3);
        if (closedLayers.length === 0) {
            state.backgroundDataUrl = state.backgroundCanvas.toDataURL("image/png");
            return;
        }
        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = state.backgroundCanvas.width;
        maskCanvas.height = state.backgroundCanvas.height;
        const maskCtx = maskCanvas.getContext("2d");
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.fillStyle = "#fff";
        closedLayers.forEach((layer) => {
            maskCtx.beginPath();
            drawPolygonPath(maskCtx, layer.polygon);
            maskCtx.fill();
        });
        const imageData = ctx.getImageData(0, 0, state.backgroundCanvas.width, state.backgroundCanvas.height);
        const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        if (state.altBackgroundImage) {
            try {
                const tmpCanvas = document.createElement("canvas");
                tmpCanvas.width = state.backgroundCanvas.width;
                tmpCanvas.height = state.backgroundCanvas.height;
                const tmpCtx = tmpCanvas.getContext("2d");
                tmpCtx.fillStyle = "#000";
                tmpCtx.fillRect(0, 0, tmpCanvas.width, tmpCanvas.height);
                tmpCtx.drawImage(
                    state.altBackgroundImage,
                    0,
                    0,
                    state.altBackgroundImage.width,
                    state.altBackgroundImage.height,
                    0,
                    0,
                    tmpCanvas.width,
                    tmpCanvas.height
                );
                const altImageData = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height);
                // Blend alt background only where mask is white.
                const altData = altImageData.data;
                const maskBuf = maskData.data;
                const baseData = imageData.data;
                for (let i = 0; i < maskBuf.length; i += 4) {
                    if (maskBuf[i] > 10) {
                        baseData[i] = altData[i];
                        baseData[i + 1] = altData[i + 1];
                        baseData[i + 2] = altData[i + 2];
                        baseData[i + 3] = 255;
                    }
                }
            } catch (err) {
                console.warn("Failed to apply alt background", err);
                inpaintImageData(imageData, maskData, { featherPasses: 2, diffusionPasses: 8 });
            }
        } else {
            inpaintImageData(imageData, maskData, { featherPasses: 2, diffusionPasses: 8 });
        }
        ctx.putImageData(imageData, 0, 0);
        state.backgroundDataUrl = state.backgroundCanvas.toDataURL("image/png");
    }

    function lockTrajectory() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        if (layer.path.length < 1) {
            setStatus("Add at least one node to animate the object.", "warn");
            return;
        }
        layer.pathLocked = true;
        updateLayerPathCache(layer);
        setStatus("Trajectory ready. Preview or export the mask.", "success");
        updateBadge();
        updateActionAvailability();
    }

    function undoTrajectoryPoint() {
        const layer = getActiveLayer();
        if (!layer || layer.path.length === 0) {
            return;
        }
        layer.path.pop();
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
    }

    function resetTrajectory() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        layer.path = [];
        layer.pathLocked = false;
        layer.selectedPathIndex = -1;
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
    }

    function handleSceneFpsChange() {
        if (!dom.sceneFpsInput) {
            return;
        }
        const value = clamp(Number(dom.sceneFpsInput.value) || state.scene.fps, 1, 240);
        state.scene.fps = value;
        dom.sceneFpsInput.value = value;
    }

    function handleSceneFrameCountChange() {
        if (!dom.sceneFrameCountInput) {
            return;
        }
        const value = clamp(parseInt(dom.sceneFrameCountInput.value, 10) || state.scene.totalFrames, 1, 5000);
        // Only adjust and sync if the value actually changed.
        if (value !== state.scene.totalFrames) {
            state.scene.totalFrames = value;
            dom.sceneFrameCountInput.value = value;
            state.layers.forEach((layer) => {
                layer.startFrame = clamp(layer.startFrame ?? 0, 0, value);
                layer.endFrame = clamp(layer.endFrame ?? value, layer.startFrame, value);
            });
            updateLayerAnimationInputs();
        }
    }

    function syncSceneInputs() {
        if (dom.sceneFpsInput) {
            dom.sceneFpsInput.value = state.scene.fps;
        }
        if (dom.sceneFrameCountInput) {
            dom.sceneFrameCountInput.value = state.scene.totalFrames;
        }
    }

    function handleLayerAnimInput(key, rawValue) {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        switch (key) {
            case "scaleStart":
                layer.scaleStart = clamp(Number(rawValue) || 1, 0.1, 3);
                break;
            case "scaleEnd":
                layer.scaleEnd = clamp(Number(rawValue) || 1, 0.1, 3);
                break;
            case "rotationStart":
                layer.rotationStart = clamp(Number(rawValue) || 0, -360, 360);
                break;
            case "rotationEnd":
                layer.rotationEnd = clamp(Number(rawValue) || 0, -360, 360);
                break;
            case "speedMode":
                layer.speedMode = rawValue || "none";
                break;
            case "speedRatio":
                layer.speedRatio = clamp(Number(rawValue) || 1, 1, 100);
                break;
            case "tension":
                layer.tension = clamp(Number(rawValue) || 0, 0, 1);
                updateLayerPathCache(layer);
                break;
            default:
                break;
        }
        updateLayerAnimationInputs();
    }

    function handleLayerFrameInput(key, rawValue) {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        const total = state.scene.totalFrames;
        if (key === "startFrame") {
            const value = clamp(Number(rawValue) || 0, 0, total);
            layer.startFrame = Math.min(value, layer.endFrame ?? total);
        } else if (key === "endFrame") {
            const value = clamp(Number(rawValue) || total, 0, total);
            layer.endFrame = Math.max(value, layer.startFrame ?? 0);
        }
        layer.startFrame = clamp(layer.startFrame ?? 0, 0, total);
        layer.endFrame = clamp(layer.endFrame ?? total, layer.startFrame, total);
        updateLayerAnimationInputs();
    }

    function handleHideOutsideRangeToggle(enabled) {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        layer.hideOutsideRange = Boolean(enabled);
        updateLayerAnimationInputs();
    }

    function updateSpeedControlVisibility(layer = getActiveLayer()) {
        if (!dom.speedModeSelect || !dom.speedRatioRow) {
            return;
        }
        const mode = layer?.speedMode || dom.speedModeSelect.value || "none";
        const hide = mode === "none";
        dom.speedRatioRow.classList.toggle("is-hidden", hide);
        if (dom.speedRatioLabel) {
            dom.speedRatioLabel.classList.toggle("is-hidden", hide);
        }
        if (dom.speedRatioInput) {
            dom.speedRatioInput.disabled = hide || !layer;
        }
    }

    function updateSpeedRatioLabel() {
        if (!dom.speedRatioValue || !dom.speedRatioInput) {
            return;
        }
        const value = Number(dom.speedRatioInput.value) || 1;
        dom.speedRatioValue.textContent = `${value.toFixed(0)}x`;
    }

    function updateCanvasPlaceholder() {
        if (!dom.canvasPlaceholder) {
            return;
        }
        const show = !state.baseImage;
        dom.canvasPlaceholder.classList.toggle("visible", show);
        if (dom.canvasFooter) {
            dom.canvasFooter.classList.toggle("hidden-controls", show);
        }
    }

    function handleClassicOutlineChange(value) {
        const numeric = clamp(Number(value) || DEFAULT_CLASSIC_OUTLINE_WIDTH, 0.5, 10);
        state.classicOutlineWidth = numeric;
        updateClassicOutlineControls();
    }

    function updateClassicOutlineControls() {
        if (!dom.classicOutlineControl) {
            return;
        }
        const width = getClassicOutlineWidth();
        const isClassic = state.renderMode === RENDER_MODES.CLASSIC;
        dom.classicOutlineControl.classList.toggle("is-hidden", !isClassic);
        if (dom.classicOutlineSlider) {
            dom.classicOutlineSlider.value = String(width);
            dom.classicOutlineSlider.disabled = !isClassic;
        }
        updateClassicOutlineLabel(width);
    }

    function updateClassicOutlineLabel(widthOverride) {
        if (!dom.classicOutlineValue) {
            return;
        }
        const width = typeof widthOverride === "number" ? widthOverride : getClassicOutlineWidth();
        dom.classicOutlineValue.textContent = `${width.toFixed(1)}px`;
    }

    function promptUnloadScene(evt) {
        evt?.preventDefault();
        if (!dom.unloadOverlay) {
            handleUnloadScene();
            return;
        }
        dom.unloadOverlay.classList.remove("hidden");
        setModalOpen(true);
    }

    function hideUnloadPrompt() {
        dom.unloadOverlay?.classList.add("hidden");
        setModalOpen(false);
    }

    function handleUnloadScene() {
        if (!state.baseImage && state.layers.every((layer) => !layer.polygonClosed)) {
            return;
        }
        state.baseImage = null;
        state.backgroundDataUrl = null;
        state.baseCanvasDataUrl = null;
        const baseCtx = state.baseCanvas.getContext("2d");
        baseCtx?.clearRect(0, 0, state.baseCanvas.width, state.baseCanvas.height);
        const bgCtx = state.backgroundCanvas.getContext("2d");
        bgCtx?.clearRect(0, 0, state.backgroundCanvas.width, state.backgroundCanvas.height);
        if (dom.sourceChooser) {
            dom.sourceChooser.value = "";
        }
        resetLayers();
        updateCanvasPlaceholder();
        setStatus("Scene unloaded. Upload a new image to continue.", "warn");
        hideUnloadPrompt();
    }

    function handleTimelineScrub(event) {
        const value = clamp(Number(event.target.value) || 0, 0, 1);
        state.animation.playhead = value;
        if (state.animation.playing) {
            state.animation.playing = false;
            dom.playPauseBtn.textContent = "Play";
        }
    }

    function togglePlayback() {
        if (state.layers.length === 0) {
            return;
        }
        state.animation.playing = !state.animation.playing;
        dom.playPauseBtn.textContent = state.animation.playing ? "Pause" : "Play";
        if (state.animation.playing) {
            state.animation.lastTime = performance.now();
            requestAnimationFrame(previewTick);
        }
    }

    function togglePreviewMode() {
        state.previewMode = !state.previewMode;
        updatePreviewModeUI();
    }

    function updatePreviewModeUI() {
        if (!dom.previewModeBtn) {
            return;
        }
        dom.previewModeBtn.textContent = state.previewMode ? "Exit Preview" : "Mask Preview";
        if (state.previewMode) {
            dom.previewModeBtn.classList.add("active");
        } else {
            dom.previewModeBtn.classList.remove("active");
        }
    }

    function previewTick(timestamp) {
        if (!state.animation.playing) {
            return;
        }
        const delta = (timestamp - state.animation.lastTime) / 1000;
        state.animation.lastTime = timestamp;
        const durationSeconds = Math.max(state.scene.totalFrames / Math.max(state.scene.fps, 1), 0.01);
        state.animation.playhead += delta / durationSeconds;
        if (state.animation.playhead >= 1) {
            if (state.animation.loop) {
                state.animation.playhead = state.animation.playhead % 1;
            } else {
                state.animation.playhead = 1;
                state.animation.playing = false;
                dom.playPauseBtn.textContent = "Play";
            }
        }
        dom.timelineSlider.value = state.animation.playhead.toFixed(3);
        requestAnimationFrame(previewTick);
    }
    function onPointerDown(evt) {
        if (evt.button !== 0) {
            return;
        }
        if (!state.baseImage) {
            dom.sourceChooser?.click();
            return;
        }
        const pt = canvasPointFromEvent(evt);
        let layer = getActiveLayer();
        let stage = getLayerStage(layer);
        const preferredLayerId = stage === "trajectory" && layer ? layer.id : null;
        const globalPathHandle = findPathHandleAtPoint(pt, 12, preferredLayerId);
        if (globalPathHandle) {
            if (!layer || globalPathHandle.layerId !== layer.id) {
                setActiveLayer(globalPathHandle.layerId);
                layer = getActiveLayer();
                stage = getLayerStage(layer);
            }
            const targetLayer = getActiveLayer();
            if (targetLayer) {
                ensureTrajectoryEditable(targetLayer);
                targetLayer.selectedPathIndex = globalPathHandle.index;
                state.dragContext = { type: "path", layerId: targetLayer.id, index: globalPathHandle.index };
            }
            evt.preventDefault();
            return;
        }
        const pathEdgeHit =
            stage === "trajectory" && layer && findBaseSegmentIndex(layer, pt, 10) !== -1 ? true : false;
        const hitLayer = pathEdgeHit ? layer : pickLayerFromPoint(pt);
        layer = getActiveLayer();
        let selectionChanged = false;
        stage = getLayerStage(layer);
        const inCreationMode = stage === "polygon";
        if (hitLayer) {
            if (!layer || hitLayer.id !== layer.id) {
                if (!inCreationMode) {
                    setActiveLayer(hitLayer.id);
                    layer = getActiveLayer();
                    stage = getLayerStage(layer);
                    selectionChanged = true;
                }
            } else {
                ensureTrajectoryEditable(layer);
                stage = getLayerStage(layer);
            }
        } else if (layer && stage === "locked") {
            setActiveLayer(null);
            layer = null;
            stage = "none";
            selectionChanged = true;
        }
        if (!layer) {
            const newLayer = addLayer(state.shapeMode);
            setActiveLayer(newLayer.id);
            layer = newLayer;
            stage = getLayerStage(layer);
            selectionChanged = false;
        }
        if (selectionChanged && layer && layer.polygonClosed) {
            enterTrajectoryCreationMode(layer, true);
            stage = getLayerStage(layer);
        }
        ensureTrajectoryEditable(layer);
        layer.shapeType = layer.shapeType || state.shapeMode || "polygon";
        stage = getLayerStage(layer);
        if (!layer.polygonClosed) {
            let handledShape = false;
            if (layer.shapeType === "rectangle") {
                handledShape = handleRectangleDrawing(layer, pt);
            } else if (layer.shapeType === "circle") {
                handledShape = handleCircleDrawing(layer, pt);
            }
            if (handledShape) {
                evt.preventDefault();
                return;
            }
        }
        state.dragContext = null;
        const polygonHandle = getPolygonHandleAtPoint(layer, pt, 10);
        if (polygonHandle) {
            layer.selectedPolygonIndex = polygonHandle.index;
            state.dragContext = {
                type: "polygon",
                layerId: layer.id,
                index: polygonHandle.index,
                modified: false,
                handleRole: polygonHandle.role || "corner",
                edgeIndex: polygonHandle.edgeIndex ?? null,
            };
            if (layer.shapeType === "circle" && layer.shapeMeta?.center) {
                state.dragContext.circleCenter = { x: layer.shapeMeta.center.x, y: layer.shapeMeta.center.y };
            }
            evt.preventDefault();
            return;
        }
        if (!layer.polygonClosed) {
            addPolygonPoint(pt);
            evt.preventDefault();
            return;
        }
        const pathIndex = findHandle(layer.path, pt, 12);
        if (pathIndex !== -1) {
            layer.selectedPathIndex = pathIndex;
            state.dragContext = { type: "path", layerId: layer.id, index: pathIndex };
            evt.preventDefault();
            return;
        }
        const canDragLayer = hitLayer && hitLayer.id === layer.id && stage !== "polygon";
        if (canDragLayer) {
            state.dragContext = {
                type: "layer",
                layerId: layer.id,
                start: pt,
                polygonSnapshot: layer.polygon.map((point) => ({ x: point.x, y: point.y })),
                pathSnapshot: layer.path.map((point) => ({ x: point.x, y: point.y })),
                shapeMetaSnapshot: cloneShapeMeta(layer.shapeMeta),
                moved: false,
            };
            evt.preventDefault();
            return;
        }
        if (!layer.pathLocked) {
            if (!selectionChanged) {
                addTrajectoryPoint(pt);
            }
            evt.preventDefault();
        }
    }

    function onPointerMove(evt) {
        if (!state.baseImage) {
            return;
        }
        const layer = getActiveLayer();
        const pt = canvasPointFromEvent(evt);
        if (state.dragContext && layer && state.dragContext.layerId === layer.id) {
            if (state.dragContext.type === "polygon") {
                if (layer.shapeType === "rectangle" && layer.polygon.length >= 4) {
                    updateRectangleHandleDrag(layer, state.dragContext.index, pt, state.dragContext);
                    state.dragContext.modified = true;
                } else if (layer.shapeType === "circle" && layer.shapeMeta?.center) {
                    updateCircleHandleDrag(layer, pt, state.dragContext);
                    state.dragContext.modified = true;
                } else if (layer.polygon[state.dragContext.index]) {
                    const prev = layer.polygon[(state.dragContext.index - 1 + layer.polygon.length) % layer.polygon.length];
                    layer.polygon[state.dragContext.index] = applyAxisLock(prev, pt, evt.shiftKey);
                    state.dragContext.modified = true;
                }
            } else if (state.dragContext.type === "path" && layer.path[state.dragContext.index]) {
                const prev = layer.path[(state.dragContext.index - 1 + layer.path.length) % layer.path.length];
                layer.path[state.dragContext.index] = applyAxisLock(prev, pt, evt.shiftKey);
                updateLayerPathCache(layer);
            } else if (state.dragContext.type === "layer" && layer.polygonClosed) {
                const dx = pt.x - state.dragContext.start.x;
                const dy = pt.y - state.dragContext.start.y;
                if (dx === 0 && dy === 0 && !state.dragContext.moved) {
                    evt.preventDefault();
                    return;
                }
                if (dx !== 0 || dy !== 0) {
                    state.dragContext.moved = true;
                }
                layer.polygon = state.dragContext.polygonSnapshot.map((point) => ({
                    x: point.x + dx,
                    y: point.y + dy,
                }));
                layer.path = state.dragContext.pathSnapshot.map((point) => ({
                    x: point.x + dx,
                    y: point.y + dy,
                }));
                if (state.dragContext.shapeMetaSnapshot) {
                    layer.shapeMeta = translateShapeMeta(state.dragContext.shapeMetaSnapshot, dx, dy);
                }
                updateLayerPathCache(layer);
            }
            evt.preventDefault();
        } else if (layer && !layer.polygonClosed) {
            if (layer.shapeType === "rectangle") {
                updateRectanglePreview(layer, pt);
            } else if (layer.shapeType === "circle") {
                updateCirclePreview(layer, pt);
            } else {
                layer.tempPolygon = null;
                layer.polygonPreviewPoint = pt;
            }
        }
    }

    function onPointerUp() {
        if (state.dragContext) {
            const layer = getLayerById(state.dragContext.layerId);
            if (
                state.dragContext.type === "layer" &&
                state.dragContext.moved &&
                layer &&
                layer.polygonClosed &&
                layer.polygon.length >= 3
            ) {
                computeLayerAssets(layer);
            } else if (
                state.dragContext.type === "polygon" &&
                state.dragContext.modified &&
                layer &&
                layer.polygonClosed &&
                layer.polygon.length >= 3
            ) {
                computeLayerAssets(layer);
            }
        }
        state.dragContext = null;
    }

    function onCanvasDoubleClick(evt) {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        const point = canvasPointFromEvent(evt);
        if (removePolygonPoint(layer, point)) {
            return;
        }
        if (removePathPoint(layer, point)) {
            return;
        }
        if (insertPointOnPolygonEdge(layer, point)) {
            return;
        }
        if (insertPointOnPathEdge(layer, point)) {
            return;
        }
        if (!layer.polygonClosed && layer.polygon.length >= 3) {
            finishPolygon();
        } else if (layer.polygonClosed && layer.path.length >= 1 && !layer.pathLocked) {
            lockTrajectory();
        }
    }

    function onCanvasContextMenu(evt) {
        evt.preventDefault();
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        if (!layer.polygonClosed) {
            if (layer.polygon.length >= 3) {
                finishPolygon();
                state.dragContext = null;
                return;
            }
            if (layer.polygon.length > 0) {
                resetPolygon();
                setStatus("Polygon creation cancelled.", "warn");
            }
            if (isLayerEmpty(layer)) {
                removeLayer();
            } else {
                setActiveLayer(null);
            }
            state.dragContext = null;
            return;
        }
        const pathCount = Array.isArray(layer.path) ? layer.path.length : 0;
        if (pathCount === 0) {
            layer.pathLocked = true;
            setActiveLayer(null);
            state.dragContext = null;
            updateBadge();
            updateActionAvailability();
            return;
        }
        if (!layer.pathLocked) {
            lockTrajectory();
            setActiveLayer(null);
            state.dragContext = null;
            return;
        }
        setActiveLayer(null);
        state.dragContext = null;
    }

    function getLayerStage(layer) {
        if (!layer) {
            return "none";
        }
        if (!layer.polygonClosed) {
            return "polygon";
        }
        if (!layer.pathLocked) {
            return "trajectory";
        }
        return "locked";
    }

    function ensureTrajectoryEditable(layer) {
        if (!layer || !layer.polygonClosed) {
            return;
        }
        if (layer.pathLocked && (!Array.isArray(layer.path) || layer.path.length <= 1)) {
            layer.pathLocked = false;
        }
    }

    function enterTrajectoryCreationMode(layer, resetSelection = false) {
        if (!layer || !layer.polygonClosed) {
            return;
        }
        if (!Array.isArray(layer.path)) {
            layer.path = [];
        }
        layer.pathLocked = false;
        if (resetSelection) {
            layer.selectedPathIndex = -1;
        }
    }

    function handleContextDelete() {
        if (!getActiveLayer()) {
            return;
        }
        removeLayer();
    }

    function handleContextReset() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        const stage = getLayerStage(layer);
        if (stage === "polygon") {
            resetPolygon();
        } else {
            resetTrajectory();
        }
    }

    function handleContextUndo() {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        const stage = getLayerStage(layer);
        if (stage === "polygon") {
            undoPolygonPoint();
        } else if (stage === "trajectory") {
            undoTrajectoryPoint();
        }
    }

    function applyAxisLock(reference, target, enable) {
        if (!enable || !reference) {
            return { x: target.x, y: target.y };
        }
        const dx = Math.abs(target.x - reference.x);
        const dy = Math.abs(target.y - reference.y);
        if (dx > dy) {
            return { x: target.x, y: reference.y };
        }
        return { x: reference.x, y: target.y };
    }

    function addPolygonPoint(pt) {
        const layer = getActiveLayer();
        if (!layer) {
            return;
        }
        layer.polygon.push(pt);
        layer.polygonPreviewPoint = null;
        updateBadge();
        updateActionAvailability();
    }

    function addTrajectoryPoint(pt) {
        const layer = getActiveLayer();
        if (!layer || !layer.polygonClosed) {
            return;
        }
        layer.path.push(pt);
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
    }

    function handleRectangleDrawing(layer, point) {
        if (!layer || layer.shapeType !== "rectangle") {
            return false;
        }
        if (!layer.shapeDraft || layer.shapeDraft.type !== "rectangle") {
            layer.shapeDraft = { type: "rectangle", start: { x: point.x, y: point.y } };
            layer.tempPolygon = null;
            layer.polygonPreviewPoint = null;
            return true;
        }
        const start = layer.shapeDraft.start;
        if (!start) {
            layer.shapeDraft = { type: "rectangle", start: { x: point.x, y: point.y } };
            return true;
        }
        if (Math.abs(point.x - start.x) < 1 && Math.abs(point.y - start.y) < 1) {
            return true;
        }
        const rectPoints = buildRectanglePoints(start, point);
        if (rectPoints.length < 4) {
            return true;
        }
        layer.polygon = rectPoints;
        layer.tempPolygon = null;
        layer.shapeDraft = null;
        layer.polygonPreviewPoint = null;
        layer.shapeMeta = {
            type: "rectangle",
            bounds: rectangleBoundsFromPoints(rectPoints),
        };
        finishPolygon();
        return true;
    }

    function handleCircleDrawing(layer, point) {
        if (!layer || layer.shapeType !== "circle") {
            return false;
        }
        if (!layer.shapeDraft || layer.shapeDraft.type !== "circle") {
            layer.shapeDraft = { type: "circle", center: { x: point.x, y: point.y } };
            layer.tempPolygon = null;
            layer.polygonPreviewPoint = null;
            return true;
        }
        const center = layer.shapeDraft.center;
        if (!center) {
            layer.shapeDraft = { type: "circle", center: { x: point.x, y: point.y } };
            return true;
        }
        const radius = Math.max(distance(center, point), 4);
        layer.polygon = buildCirclePolygon(center, radius);
        layer.tempPolygon = null;
        layer.shapeDraft = null;
        layer.polygonPreviewPoint = null;
        layer.shapeMeta = {
            type: "circle",
            center: { x: center.x, y: center.y },
            radius,
        };
        finishPolygon();
        return true;
    }

    function updateRectanglePreview(layer, point) {
        if (!layer || layer.shapeType !== "rectangle" || !layer.shapeDraft || layer.shapeDraft.type !== "rectangle") {
            return;
        }
        layer.polygonPreviewPoint = null;
        layer.tempPolygon = buildRectanglePoints(layer.shapeDraft.start, point);
    }

    function updateCirclePreview(layer, point) {
        if (!layer || layer.shapeType !== "circle" || !layer.shapeDraft || layer.shapeDraft.type !== "circle") {
            return;
        }
        const center = layer.shapeDraft.center;
        if (!center) {
            return;
        }
        layer.polygonPreviewPoint = null;
        const radius = Math.max(distance(center, point), 2);
        layer.tempPolygon = buildCirclePolygon(center, radius);
    }

    function updateBadge() {
        updateWorkflowPanels();
        if (!dom.badge) {
            return;
        }
        if (!state.baseImage) {
            dom.badge.textContent = "";
            dom.badge.style.display = "none";
            return;
        }
        dom.badge.style.display = "block";
        const layer = getActiveLayer();
        if (!layer) {
            dom.badge.textContent = "Left click anywhere on the canvas to start a new object.";
            dom.badge.style.display = "block";
            return;
        }
        if (!layer.polygonClosed) {
            dom.badge.textContent = "Left click to draw the polygon. Double click to split edges. Right click to close.";
            dom.badge.style.display = "block";
            return;
        }
        if (layer.path.length < 1) {
            dom.badge.textContent = "Add trajectory points with left click. Right click to lock the path.";
            dom.badge.style.display = "block";
            return;
        }
        if (!layer.pathLocked) {
            dom.badge.textContent = "Continue refining the trajectory or right click to lock it.";
            dom.badge.style.display = "block";
            return;
        }
        dom.badge.textContent = "Adjust animation ranges or start another object.";
        dom.badge.style.display = "block";
    }

    function updateActionAvailability() {
        const layer = getActiveLayer();
        updateContextButtons(layer);
    }

    function updateContextButtons(layer) {
        if (!dom.contextDeleteBtn || !dom.contextResetBtn || !dom.contextUndoBtn) {
            return;
        }
        if (dom.contextButtonGroup) {
            dom.contextButtonGroup.classList.toggle("is-hidden", !layer);
        }
        if (!layer) {
            dom.contextDeleteBtn.disabled = true;
            dom.contextResetBtn.disabled = true;
            dom.contextUndoBtn.disabled = true;
            dom.contextDeleteBtn.textContent = "Delete";
            dom.contextResetBtn.textContent = "Reset";
            dom.contextUndoBtn.textContent = "Undo";
            return;
        }
        const stage = getLayerStage(layer);
        dom.contextDeleteBtn.disabled = false;
        dom.contextDeleteBtn.textContent = "Delete Object";
        if (stage === "polygon") {
            dom.contextResetBtn.disabled = layer.polygon.length === 0;
            dom.contextUndoBtn.disabled = layer.polygon.length === 0;
            dom.contextResetBtn.textContent = "Reset Polygon";
            dom.contextUndoBtn.textContent = "Undo Point";
        } else if (stage === "trajectory") {
            dom.contextResetBtn.disabled = layer.path.length === 0;
            dom.contextUndoBtn.disabled = layer.path.length === 0;
            dom.contextResetBtn.textContent = "Reset Trajectory";
            dom.contextUndoBtn.textContent = "Undo Point";
        } else {
            dom.contextResetBtn.disabled = false;
            dom.contextUndoBtn.disabled = true;
            dom.contextResetBtn.textContent = "Reset Trajectory";
            dom.contextUndoBtn.textContent = "Undo Point";
        }
    }

    function expandPanel(panelId, userInitiated = false) {
        if (!dom.collapsiblePanels) {
            return;
        }
        dom.collapsiblePanels.forEach((panel) => {
            const isMatch = panel.dataset.panel === panelId;
            if (panel.dataset.fixed === "true") {
                panel.classList.add("expanded");
            } else {
                panel.classList.toggle("expanded", isMatch);
            }
        });
        state.currentExpandedPanel = panelId;
        if (userInitiated) {
            state.userPanelOverride = panelId;
        } else if (!state.userPanelOverride) {
            state.userPanelOverride = null;
        }
    }

    function getWorkflowStage() {
        if (!state.baseImage) {
            return "scene";
        }
        return "object";
    }

    function updateWorkflowPanels() {
        if (!dom.collapsiblePanels || dom.collapsiblePanels.length === 0) {
            return;
        }
        const stage = getWorkflowStage();
        if (state.currentWorkflowStage !== stage) {
            state.currentWorkflowStage = stage;
            state.userPanelOverride = null;
        }
        if (!state.userPanelOverride) {
            expandPanel(stage);
        }
    }

    function layerReadyForExport(layer) {
        if (!layer) {
            return false;
        }
        if (!layer.objectCut || !layer.polygonClosed) {
            return false;
        }
        // Allow static objects (no trajectory) as long as their shape is finalized.
        return true;
    }

    function downloadBackgroundImage() {
        if (!state.backgroundDataUrl) {
            setStatus("Background image not ready yet.", "warn");
            return;
        }
        const link = document.createElement("a");
        link.href = state.backgroundDataUrl;
        link.download = "motion_designer_background.png";
        link.click();
    }

    function getExportBackgroundImage(renderMode = state.renderMode) {
        return state.baseCanvasDataUrl;
    }

    function resetTransferState() {
        state.transfer.pending = false;
        state.transfer.mask = null;
        state.transfer.guide = null;
        state.transfer.backgroundImage = null;
    }

    function handleSendToWangp() {
        if (state.renderMode === RENDER_MODES.CUT_DRAG) {
            state.transfer.pending = true;
            state.transfer.mask = null;
            state.transfer.guide = null;
            state.transfer.backgroundImage = getExportBackgroundImage();
        } else {
            resetTransferState();
        }
        setStatus("Preparing motion mask for WanGP...", "info");
        const started = startExport("wangp", "mask");
        if (!started && state.renderMode === RENDER_MODES.CUT_DRAG) {
            resetTransferState();
        }
    }

    function startExport(mode, variant = "mask") {
        const readyLayers = state.layers.filter((layer) => layerReadyForExport(layer));
        if (readyLayers.length === 0) {
            setStatus("Prepare at least one object and trajectory before exporting.", "warn");
            return false;
        }
        const mimeType = pickSupportedMimeType();
        if (!mimeType) {
            notify("MediaRecorder with VP8/VP9 support is required to export the mask.", "error");
            return false;
        }
        if (state.export.running) {
            return false;
        }
        const totalFrames = Math.max(1, Math.round(state.scene.totalFrames));
        state.export.running = true;
        state.export.mode = mode;
        state.export.variant = variant === "guide" ? "guide" : "mask";
        state.export.renderMode = state.renderMode;
        state.export.frame = 0;
        state.export.totalFrames = totalFrames;
        state.export.mimeType = mimeType;
        state.export.chunks = [];
        state.export.cancelled = false;
        state.export.canvas = document.createElement("canvas");
        state.export.canvas.width = state.resolution.width;
        state.export.canvas.height = state.resolution.height;
        state.export.ctx = state.export.canvas.getContext("2d", { alpha: false });
        state.export.stream = state.export.canvas.captureStream(Math.max(state.scene.fps, 1));
        state.export.track = state.export.stream.getVideoTracks()[0] || null;
        const pixelCount = state.resolution.width * state.resolution.height;
        const targetFps = Math.max(state.scene.fps, 1);
        const targetBitrate = Math.round(
            Math.min(16_000_000, Math.max(3_000_000, pixelCount * targetFps * 0.12))
        );
        state.export.recorder = new MediaRecorder(state.export.stream, { mimeType, videoBitsPerSecond: targetBitrate });
        state.export.recorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                state.export.chunks.push(event.data);
            }
        };
        state.export.recorder.onstop = finalizeExport;
        const timeslice = Math.max(10, Math.round(1000 / targetFps));
        state.export.recorder.start(timeslice);
        setModalOpen(true);
        dom.exportOverlay?.classList.remove("hidden");
        dom.exportProgressBar.style.width = "0%";
        const labelPrefix = state.export.variant === "guide" ? "Rendering Control Frame" : "Rendering Mask Frame";
        dom.exportLabel.textContent = `${labelPrefix} 0 / ${totalFrames}`;
        renderNextExportFrame();
        return true;
    }
    function cancelExport() {
        if (!state.export.running) {
            dom.exportOverlay?.classList.add("hidden");
            setModalOpen(false);
            return;
        }
        state.export.cancelled = true;
        try {
            state.export.recorder.stop();
        } catch (err) {
            console.warn("Failed stopping recorder", err);
        }
        dom.exportOverlay?.classList.add("hidden");
        state.export.running = false;
        resetTransferState();
        setStatus("Export cancelled.", "warn");
        setModalOpen(false);
    }

    function renderNextExportFrame() {
        if (!state.export.running || state.export.cancelled) {
            return;
        }
        if (state.export.frame >= state.export.totalFrames) {
            state.export.running = false;
            const frameDelayMs = 1000 / Math.max(state.scene.fps, 1);
            const recorder = state.export.recorder;
            if (recorder && recorder.state === "recording") {
                try {
                    // Flush any buffered data before stopping to avoid losing the last frame.
                    recorder.requestData();
                } catch (err) {
                    console.warn("Unable to request final data chunk", err);
                }
                setTimeout(() => {
                    try {
                        if (recorder.state === "recording") {
                            recorder.stop();
                        }
                    } catch (err) {
                        console.warn("Unable to stop recorder", err);
                    }
                }, Math.max(50, frameDelayMs * 1.5));
            } else {
                try {
                    recorder?.stop();
                } catch (err) {
                    console.warn("Unable to stop recorder", err);
                }
            }
            return;
        }
        const progress = state.export.totalFrames === 1 ? 0 : state.export.frame / (state.export.totalFrames - 1);
        if (state.export.variant === "guide") {
            drawGuideFrame(state.export.ctx, progress);
        } else {
            drawMaskFrame(state.export.ctx, progress);
        }
        if (state.export.track && typeof state.export.track.requestFrame === "function") {
            try {
                state.export.track.requestFrame();
            } catch (err) {
                console.warn("requestFrame failed", err);
            }
        }
        const percentage = ((state.export.frame + 1) / state.export.totalFrames) * 100;
        dom.exportProgressBar.style.width = `${percentage}%`;
        const labelPrefix = state.export.variant === "guide" ? "Rendering Control Frame" : "Rendering Mask Frame";
        dom.exportLabel.textContent = `${labelPrefix} ${state.export.frame + 1} / ${state.export.totalFrames}`;
        state.export.frame += 1;
        const frameDelay = 1000 / Math.max(state.scene.fps, 1);
        setTimeout(renderNextExportFrame, frameDelay);
    }

    function serializeLayer(layer) {
        if (!layer) {
            return null;
        }
        const clonePoints = (pts) => (Array.isArray(pts) ? pts.map((p) => ({ x: p.x, y: p.y })) : []);
        const serializeShapeMeta = (meta) => {
            if (!meta) return null;
            if (meta.type === "rectangle" && meta.bounds) {
                return { type: "rectangle", bounds: { ...meta.bounds } };
            }
            if (meta.type === "circle" && meta.center) {
                return { type: "circle", center: { ...meta.center }, radius: meta.radius };
            }
            return null;
        };
        return {
            id: layer.id,
            name: layer.name,
            color: layer.color,
            polygon: clonePoints(layer.polygon),
            polygonClosed: !!layer.polygonClosed,
            path: clonePoints(layer.path),
            pathLocked: !!layer.pathLocked,
            shapeType: layer.shapeType || "polygon",
            shapeMeta: serializeShapeMeta(layer.shapeMeta),
            startFrame: layer.startFrame ?? 0,
            endFrame: layer.endFrame ?? state.scene.totalFrames,
            scaleStart: layer.scaleStart ?? 1,
            scaleEnd: layer.scaleEnd ?? 1,
            rotationStart: layer.rotationStart ?? 0,
            rotationEnd: layer.rotationEnd ?? 0,
            speedMode: layer.speedMode || "none",
            speedRatio: layer.speedRatio ?? 1,
            tension: layer.tension ?? 0,
            hideOutsideRange: !!layer.hideOutsideRange,
        };
    }

    function deserializeLayer(def, colorIndex) {
        const layer = createLayer(def.name || `Object ${colorIndex + 1}`, colorIndex, def.shapeType || "polygon");
        layer.id = def.id || layer.id;
        layer.color = def.color || layer.color;
        layer.polygon = Array.isArray(def.polygon) ? def.polygon.map((p) => ({ x: p.x, y: p.y })) : [];
        layer.polygonClosed = !!def.polygonClosed;
        layer.path = Array.isArray(def.path) ? def.path.map((p) => ({ x: p.x, y: p.y })) : [];
        layer.pathLocked = !!def.pathLocked;
        layer.shapeType = def.shapeType || layer.shapeType;
        layer.shapeMeta = def.shapeMeta || null;
        layer.startFrame = clamp(def.startFrame ?? 0, 0, state.scene.totalFrames);
        layer.endFrame = clamp(def.endFrame ?? state.scene.totalFrames, layer.startFrame, state.scene.totalFrames);
        layer.scaleStart = def.scaleStart ?? 1;
        layer.scaleEnd = def.scaleEnd ?? 1;
        layer.rotationStart = def.rotationStart ?? 0;
        layer.rotationEnd = def.rotationEnd ?? 0;
        layer.speedMode = def.speedMode || "none";
        layer.speedRatio = def.speedRatio ?? 1;
        layer.tension = def.tension ?? 0;
        layer.hideOutsideRange = !!def.hideOutsideRange;
        updateLayerPathCache(layer);
        if (state.baseImage && layer.polygonClosed && layer.polygon.length >= 3) {
            computeLayerAssets(layer);
        }
        return layer;
    }

    function buildDefinition() {
        return {
            resolution: { ...state.resolution },
            fitMode: state.fitMode,
            renderMode: state.renderMode,
            altBackground: state.altBackgroundDataUrl || null,
            showPatchedBackground: state.showPatchedBackground,
            classicOutlineWidth: state.classicOutlineWidth,
            scene: { fps: state.scene.fps, totalFrames: state.scene.totalFrames },
            layers: state.layers.map(serializeLayer).filter(Boolean),
        };
    }

    function applyDefinition(def) {
        if (!def || typeof def !== "object") {
            notify("Invalid definition file.", "error");
            return;
        }
        if (!state.baseImage) {
            notify("Load a base image first, then load the definition.", "warn");
            return;
        }
        try {
            if (def.resolution && def.resolution.width && def.resolution.height) {
                state.resolution = {
                    width: clamp(Number(def.resolution.width) || state.resolution.width, 128, 4096),
                    height: clamp(Number(def.resolution.height) || state.resolution.height, 128, 4096),
                };
                if (dom.targetWidthInput) dom.targetWidthInput.value = state.resolution.width;
                if (dom.targetHeightInput) dom.targetHeightInput.value = state.resolution.height;
                matchCanvasResolution();
            }
            if (def.fitMode) {
                state.fitMode = def.fitMode;
                if (dom.fitModeSelect) dom.fitModeSelect.value = state.fitMode;
                applyResolution();
            }
            if (def.scene) {
                state.scene.fps = clamp(Number(def.scene.fps) || state.scene.fps, 1, 240);
                state.scene.totalFrames = clamp(Number(def.scene.totalFrames) || state.scene.totalFrames, 1, 5000);
                syncSceneInputs();
            }
            if (def.renderMode) {
                setRenderMode(def.renderMode);
            }
            if (def.altBackground) {
                loadAltBackground(def.altBackground);
            } else {
                clearAltBackground(false);
            }
            if (typeof def.showPatchedBackground === "boolean") {
                state.showPatchedBackground = def.showPatchedBackground;
            }
            if (typeof def.classicOutlineWidth === "number") {
                state.classicOutlineWidth = def.classicOutlineWidth;
            }
            state.layers = [];
            const layers = Array.isArray(def.layers) ? def.layers : [];
            layers.forEach((ldef, idx) => {
                const layer = deserializeLayer(ldef, idx);
                state.layers.push(layer);
            });
            state.activeLayerId = state.layers[0]?.id || null;
            recomputeBackgroundFill();
            refreshLayerSelect();
            updateLayerAnimationInputs();
            updateBadge();
            updateActionAvailability();
            updateCanvasPlaceholder();
            updateClassicOutlineControls();
        } catch (err) {
            console.warn("Failed to apply definition", err);
            notify("Failed to load definition.", "error");
        }
    }

    function handleSaveDefinition() {
        const def = buildDefinition();
        const blob = new Blob([JSON.stringify(def, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "motion_designer_definition.json";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    function handleDefinitionFileSelected(event) {
        const file = event.target?.files?.[0];
        if (!file) {
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const parsed = JSON.parse(String(reader.result || "{}"));
                applyDefinition(parsed);
            } catch (err) {
                console.warn("Failed to parse definition", err);
                notify("Invalid definition file.", "error");
            }
        };
        reader.readAsText(file);
    }

    function loadAltBackground(dataUrl) {
        if (!dataUrl) {
            clearAltBackground(false);
            return;
        }
        const img = new Image();
        img.onload = () => {
            state.altBackgroundImage = img;
            state.altBackgroundDataUrl = dataUrl;
            state.showPatchedBackground = true;
            updateBackgroundToggleUI();
            recomputeBackgroundFill();
        };
        img.onerror = () => {
            clearAltBackground(true);
        };
        img.src = dataUrl;
    }

    function clearAltBackground(shouldAlert = false) {
        state.altBackgroundImage = null;
        state.altBackgroundDataUrl = null;
        updateBackgroundToggleUI();
        recomputeBackgroundFill();
        if (shouldAlert) {
            notify("Alternative background failed to load. Restoring inpainted fill.", "warn");
        }
    }

    function handleBackgroundFileSelected(event) {
        const file = event.target?.files?.[0];
        if (!file) {
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            loadAltBackground(String(reader.result || ""));
        };
        reader.readAsDataURL(file);
    }

    function handleBackgroundToggle(evt) {
        evt?.preventDefault();
        if (state.renderMode !== RENDER_MODES.CUT_DRAG) {
            notify("Setting an alternate background is only available in Cut & Drag mode.", "warn");
            return;
        }
        if (!state.baseImage) {
            notify("Load a base image first, then set an alternate background.", "warn");
            return;
        }
        if (state.altBackgroundImage) {
            clearAltBackground(false);
        } else if (dom.backgroundFileInput) {
            dom.backgroundFileInput.value = "";
            dom.backgroundFileInput.click();
        }
    }

    function updateBackgroundToggleUI() {
        if (!dom.backgroundToggleBtn) {
            return;
        }
        const hasAlt = !!state.altBackgroundImage;
        const show = state.renderMode === RENDER_MODES.CUT_DRAG;
        dom.backgroundToggleBtn.classList.toggle("is-hidden", !show);
        dom.backgroundToggleBtn.textContent = hasAlt ? "Clear Background" : "Set Default Background";
    }

    function finalizeExport() {
        dom.exportOverlay?.classList.add("hidden");
        const blob = new Blob(state.export.chunks, { type: state.export.mimeType || "video/webm" });
        state.export.running = false;
        if (state.export.cancelled) {
            state.export.chunks = [];
            setModalOpen(false);
            return;
        }
        const durationSeconds = state.scene.totalFrames / Math.max(state.scene.fps, 1);
        prepareExportBlob(blob, durationSeconds);
        setModalOpen(false);
    }

    function buildExportMetadata() {
        return {
            mimeType: state.export.mimeType || "video/webm",
            width: state.resolution.width,
            height: state.resolution.height,
            fps: state.scene.fps,
            duration: state.scene.totalFrames / Math.max(state.scene.fps, 1),
            totalFrames: state.export.totalFrames,
            renderedFrames: state.export.frame,
            expectedFrames: state.export.totalFrames,
            renderMode: state.export.renderMode,
            variant: state.export.variant,
        };
    }

    async function prepareExportBlob(blob, durationSeconds) {
        const preparedBlob = blob;

        if (state.export.mode === "download") {
            const link = document.createElement("a");
            link.href = URL.createObjectURL(preparedBlob);
            link.download = "motion_designer_mask.webm";
            link.click();
            setStatus("Mask video downloaded.", "success");
            state.export.chunks = [];
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            try {
                const dataUrl = reader.result || "";
                const payload = typeof dataUrl === "string" ? dataUrl.split(",")[1] || "" : "";
                const metadata = buildExportMetadata();
                handleWanGPExportPayload(payload, metadata);
            } catch (err) {
                console.error("Unable to forward mask to WanGP", err);
                setStatus("Failed to send mask to WanGP.", "error");
                resetTransferState();
            } finally {
                state.export.chunks = [];
            }
        };
        reader.readAsDataURL(preparedBlob);
    }

    function handleWanGPExportPayload(payload, metadata) {
        if (state.export.renderMode === RENDER_MODES.CUT_DRAG && state.transfer.pending) {
            if (state.export.variant === "mask") {
                state.transfer.mask = { payload, metadata };
                if (!state.transfer.backgroundImage) {
                    state.transfer.backgroundImage = getExportBackgroundImage(state.export.renderMode);
                }
                setStatus("Mask ready. Rendering motion preview...", "info");
                startExport("wangp", "guide");
                return;
            }
            if (state.export.variant === "guide") {
                state.transfer.guide = { payload, metadata };
                if (!dispatchPendingTransfer()) {
                    console.warn("Unable to dispatch Motion Designer transfer. Missing mask or metadata.");
                }
                return;
            }
        }
        const success = sendMotionDesignerPayload({
            payload,
            metadata,
            backgroundImage: getExportBackgroundImage(state.export.renderMode),
        });
        if (success) {
            resetTransferState();
            setStatus("Mask sent to WanGP.", "success");
        }
    }

    function dispatchPendingTransfer() {
        if (!state.transfer.pending || !state.transfer.mask || !state.transfer.guide) {
            return false;
        }
        const success = sendMotionDesignerPayload({
            payload: state.transfer.mask.payload,
            metadata: state.transfer.mask.metadata,
            backgroundImage: state.transfer.backgroundImage || getExportBackgroundImage(state.export.renderMode),
            guidePayload: state.transfer.guide.payload,
            guideMetadata: state.transfer.guide.metadata,
        });
        if (success) {
            setStatus("Mask and preview sent to WanGP.", "success");
            resetTransferState();
        }
        return success;
    }

    function sendMotionDesignerPayload({ payload, metadata, backgroundImage, guidePayload = null, guideMetadata = null }) {
        try {
            window.parent?.postMessage(
                { type: EVENT_TYPE, payload, metadata, backgroundImage, guidePayload, guideMetadata },
                "*",
            );
            return true;
        } catch (err) {
            console.error("Unable to forward mask to WanGP", err);
            setStatus("Failed to send mask to WanGP.", "error");
            return false;
        }
    }

    function pickSupportedMimeType() {
        if (!window.MediaRecorder) {
            return null;
        }
        for (const mime of MIME_CANDIDATES) {
            if (MediaRecorder.isTypeSupported(mime)) {
                return mime;
            }
        }
        return null;
    }

    function drawMaskFrame(ctx, progress, options = {}) {
        const { fillBackground = false, backgroundColor = "#000", fillColor = "white" } = options;
        const currentRenderMode = state.export.running ? state.export.renderMode : state.renderMode;
        const outlineMode = currentRenderMode === RENDER_MODES.CLASSIC;
        const outlineWidth = outlineMode ? getClassicOutlineWidth() : 0;
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = backgroundColor;
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        state.layers
            .filter((layer) => layerReadyForExport(layer))
            .forEach((layer) => {
                const transform = computeTransform(layer, progress);
                if (!transform || layer.localPolygon.length === 0) {
                    return;
                }
                ctx.save();
                ctx.translate(transform.position.x, transform.position.y);
                ctx.rotate((transform.rotation * Math.PI) / 180);
                ctx.scale(transform.scale, transform.scale);
                ctx.translate(-layer.anchor.x, -layer.anchor.y);
                if (outlineMode) {
                    ctx.strokeStyle = fillColor;
                    ctx.lineWidth = outlineWidth;
                    ctx.lineJoin = "round";
                    ctx.lineCap = "round";
                    drawLocalPolygon(ctx, layer.localPolygon);
                    ctx.stroke();
                } else {
                    ctx.fillStyle = fillColor;
                    drawLocalPolygon(ctx, layer.localPolygon);
                    ctx.fill();
                }
                ctx.restore();
            });
    }

    function drawGuideFrame(ctx, progress) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        if (!state.baseImage) {
            ctx.fillStyle = state.theme === "dark" ? "#000000" : "#ffffff";
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            return;
        }
        const currentRenderMode = state.export.running ? state.export.renderMode : state.renderMode;
        const useClassic = currentRenderMode === RENDER_MODES.CLASSIC;
        const usePatchedBackground = !useClassic && state.showPatchedBackground && state.backgroundDataUrl;
        if (usePatchedBackground) {
            ctx.drawImage(state.backgroundCanvas, 0, 0);
        } else {
            ctx.drawImage(state.baseCanvas, 0, 0);
        }
        drawPreviewObjects(ctx, { progress });
    }

    function render() {
        const ctx = dom.ctx;
        if (state.previewMode) {
            drawMaskFrame(ctx, state.animation.playhead, {
                fillBackground: true,
                backgroundColor: "#04060b",
                fillColor: "white",
            });
        } else {
            ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
            if (state.baseImage) {
                const useClassic = state.renderMode === RENDER_MODES.CLASSIC;
                const usePatchedBackground = !useClassic && state.showPatchedBackground && state.backgroundDataUrl;
                if (usePatchedBackground) {
                    ctx.drawImage(state.backgroundCanvas, 0, 0);
                } else {
                    ctx.drawImage(state.baseCanvas, 0, 0);
                }
                drawPolygonOverlay(ctx);
                drawPreviewObjects(ctx);
                drawTrajectoryOverlay(ctx);
            } else {
                ctx.fillStyle = state.theme === "dark" ? "#000000" : "#ffffff";
                ctx.fillRect(0, 0, dom.canvas.width, dom.canvas.height);
            }
        }
        requestAnimationFrame(render);
    }

    function drawPolygonOverlay(ctx) {
        state.layers.forEach((layer) => {
            const points = getPolygonDisplayPoints(layer);
            if (!points || points.length === 0) {
                return;
            }
            const isClosed = layer.polygonClosed || (!layer.polygonClosed && layer.shapeType !== "polygon" && points.length >= 3);
            ctx.save();
            ctx.strokeStyle = POLYGON_EDGE_COLOR;
            ctx.globalAlpha = layer.id === state.activeLayerId ? 1 : 0.45;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i].x, points[i].y);
            }
            if (isClosed) {
                ctx.closePath();
                if (layer.polygonClosed && points.length > 2) {
                    ctx.fillStyle = "rgba(255,77,87,0.08)";
                    ctx.fill();
                }
            } else if (layer.shapeType === "polygon" && layer.polygonPreviewPoint) {
                ctx.lineTo(layer.polygonPreviewPoint.x, layer.polygonPreviewPoint.y);
            }
            ctx.stroke();
            const handles = getPolygonHandlePoints(layer, points);
            handles.forEach((handle) => {
                ctx.beginPath();
                ctx.arc(handle.x, handle.y, 6, 0, Math.PI * 2);
                ctx.fillStyle = handle.index === layer.selectedPolygonIndex ? "#ff9aa6" : "#081320";
                ctx.fill();
                ctx.strokeStyle = POLYGON_EDGE_COLOR;
                ctx.stroke();
            });
            ctx.restore();
        });
    }

    function drawTrajectoryOverlay(ctx) {
        state.layers.forEach((layer) => {
            const points = getRenderPath(layer);
            if (!points || points.length === 0) {
                return;
            }
            const showPath = points.length > 1;
            ctx.save();
            ctx.strokeStyle = TRAJECTORY_EDGE_COLOR;
            ctx.globalAlpha = layer.id === state.activeLayerId ? 0.95 : 0.5;
            ctx.lineWidth = 2;
            if (showPath) {
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }
                ctx.stroke();
            }
            // Draw anchor indicator
            ctx.beginPath();
            ctx.arc(points[0].x, points[0].y, 5, 0, Math.PI * 2);
            ctx.fillStyle = layer.id === state.activeLayerId ? "#57f0b7" : "#1a2b3d";
            ctx.fill();
            ctx.strokeStyle = TRAJECTORY_EDGE_COLOR;
            ctx.stroke();
            // Draw user nodes
            layer.path.forEach((point, index) => {
                ctx.beginPath();
                ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
                ctx.fillStyle = index === layer.selectedPathIndex ? "#ffd69b" : "#1e1507";
                ctx.fill();
                ctx.strokeStyle = "#ffbe7a";
                ctx.stroke();
            });
            ctx.restore();
        });
    }

    function getRenderPath(layer) {
        if (!layer) {
            return [];
        }
        if (Array.isArray(layer.renderPath) && layer.renderPath.length > 0) {
            return layer.renderPath;
        }
        return getEffectivePath(layer);
    }

    function getSceneFrameCount() {
        const value = Number(state.scene.totalFrames);
        return Math.max(1, Number.isFinite(value) ? value : 1);
    }

    function progressToFrame(progress) {
        return clamp(progress, 0, 1) * getSceneFrameCount();
    }

    function getLayerFrameWindow(layer) {
        const total = getSceneFrameCount();
        const rawStart = layer && typeof layer.startFrame === "number" ? layer.startFrame : 0;
        const rawEnd = layer && typeof layer.endFrame === "number" ? layer.endFrame : total;
        const start = clamp(rawStart, 0, total);
        const end = clamp(rawEnd, start, total);
        return { start, end };
    }

    function getLayerTimelineProgress(layer, timelineProgress) {
        if (!layer) {
            return null;
        }
        const { start, end } = getLayerFrameWindow(layer);
        const currentFrame = progressToFrame(timelineProgress);
        const hideOutside = layer.hideOutsideRange !== false;
        if (currentFrame < start) {
            return hideOutside ? null : 0;
        }
        if (currentFrame > end) {
            return hideOutside ? null : 1;
        }
        const span = Math.max(end - start, 0.0001);
        return clamp((currentFrame - start) / span, 0, 1);
    }

    function drawPreviewObjects(ctx, options = {}) {
        const previewProgress =
            typeof options.progress === "number" ? clamp(options.progress, 0, 1) : state.animation.playhead;
        const useClassic = state.renderMode === RENDER_MODES.CLASSIC;
        const outlineWidth = useClassic ? getClassicOutlineWidth() : 0;
        state.layers.forEach((layer) => {
            if (!layer.objectCut) {
                return;
            }
            const hasPath = Array.isArray(layer.path) && layer.path.length > 0;
            if (!layer.pathLocked && !hasPath) {
                return;
            }
            const transform = computeTransform(layer, previewProgress);
            if (!transform) {
                return;
            }
            ctx.save();
            ctx.translate(transform.position.x, transform.position.y);
            ctx.rotate((transform.rotation * Math.PI) / 180);
            ctx.scale(transform.scale, transform.scale);
            ctx.translate(-layer.anchor.x, -layer.anchor.y);
            if (useClassic) {
                if (!Array.isArray(layer.localPolygon) || layer.localPolygon.length === 0) {
                    ctx.restore();
                    return;
                }
                ctx.globalAlpha = 1;
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = outlineWidth;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                drawLocalPolygon(ctx, layer.localPolygon);
                ctx.stroke();
            } else {
                ctx.globalAlpha = 0.92;
                ctx.drawImage(layer.objectCut.canvas, 0, 0);
            }
            ctx.restore();
        });
    }

    function computeTransform(layer, progress) {
        if (!layer || !layer.objectCut) {
            return null;
        }
        const points = getRenderPath(layer);
        if (!points || points.length === 0) {
            return null;
        }
        const localProgress = getLayerTimelineProgress(layer, progress);
        if (localProgress === null) {
            return null;
        }
        const speedProgress = getSpeedProfileProgress(layer, localProgress);
        const meta = layer.pathMeta || computePathMeta(points) || { lengths: [0], total: 0 };
        let position;
        if (points.length === 1 || !meta || meta.total === 0) {
            position = points[0];
        } else {
            const target = speedProgress * meta.total;
            let idx = 0;
            while (idx < meta.lengths.length - 1 && target > meta.lengths[idx + 1]) {
                idx++;
            }
            const segStart = points[idx];
            const segEnd = points[Math.min(idx + 1, points.length - 1)];
            const segStartDist = meta.lengths[idx];
            const segLen = Math.max(meta.lengths[idx + 1] - segStartDist, 0.0001);
            const t = (target - segStartDist) / segLen;
            position = { x: lerp(segStart.x, segEnd.x, t), y: lerp(segStart.y, segEnd.y, t) };
        }
        const scaleStart = typeof layer.scaleStart === "number" ? layer.scaleStart : 1;
        const scaleEnd = typeof layer.scaleEnd === "number" ? layer.scaleEnd : 1;
        const rotationStart = typeof layer.rotationStart === "number" ? layer.rotationStart : 0;
        const rotationEnd = typeof layer.rotationEnd === "number" ? layer.rotationEnd : 0;
        const scale = lerp(scaleStart, scaleEnd, speedProgress);
        const rotation = lerp(rotationStart, rotationEnd, speedProgress);
        return { position, scale, rotation };
    }

    function getSpeedProfileProgress(layer, progress) {
        const ratio = clamp(Number(layer?.speedRatio) || 1, 1, 100);
        const mode = layer?.speedMode || "accelerate";
        const segments = buildSpeedSegments(mode, ratio);
        return evaluateSpeedSegments(clamp(progress, 0, 1), segments);
    }

    function buildSpeedSegments(mode, ratio) {
        const minSpeed = 1;
        const maxSpeed = ratio;
        switch (mode) {
            case "none":
                return [];
            case "decelerate":
                return [{ duration: 1, start: maxSpeed, end: minSpeed }];
            case "accelerate-decelerate":
                return [
                    { duration: 0.5, start: minSpeed, end: maxSpeed },
                    { duration: 0.5, start: maxSpeed, end: minSpeed },
                ];
            case "decelerate-accelerate":
                return [
                    { duration: 0.5, start: maxSpeed, end: minSpeed },
                    { duration: 0.5, start: minSpeed, end: maxSpeed },
                ];
            case "accelerate":
            default:
                return [{ duration: 1, start: minSpeed, end: maxSpeed }];
        }
    }

    function evaluateSpeedSegments(progress, segments) {
        if (!Array.isArray(segments) || segments.length === 0) {
            return progress;
        }
        const totalArea = segments.reduce((sum, seg) => sum + seg.duration * ((seg.start + seg.end) / 2), 0) || 1;
        let accumulatedTime = 0;
        let accumulatedArea = 0;
        for (const seg of segments) {
            const segEndTime = accumulatedTime + seg.duration;
            if (progress >= segEndTime) {
                accumulatedArea += seg.duration * ((seg.start + seg.end) / 2);
                accumulatedTime = segEndTime;
                continue;
            }
            const localDuration = seg.duration;
            const localT = localDuration === 0 ? 0 : (progress - accumulatedTime) / localDuration;
            const areaInSegment =
                localDuration * (seg.start * localT + 0.5 * (seg.end - seg.start) * localT * localT);
            return (accumulatedArea + areaInSegment) / totalArea;
        }
        return 1;
    }

    function getClassicOutlineWidth() {
        const width = typeof state.classicOutlineWidth === "number" ? state.classicOutlineWidth : DEFAULT_CLASSIC_OUTLINE_WIDTH;
        return clamp(width, 0.5, 10);
    }

    function buildRectanglePoints(a, b) {
        if (!a || !b) {
            return [];
        }
        const minX = Math.min(a.x, b.x);
        const maxX = Math.max(a.x, b.x);
        const minY = Math.min(a.y, b.y);
        const maxY = Math.max(a.y, b.y);
        return [
            { x: minX, y: minY },
            { x: maxX, y: minY },
            { x: maxX, y: maxY },
            { x: minX, y: maxY },
        ];
    }

    function rectangleBoundsFromPoints(points) {
        if (!Array.isArray(points) || points.length === 0) {
            return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
        }
        const bounds = polygonBounds(points);
        return {
            minX: bounds.minX,
            minY: bounds.minY,
            maxX: bounds.minX + bounds.width,
            maxY: bounds.minY + bounds.height,
        };
    }

    function buildCirclePolygon(center, radius, segments = 48) {
        if (!center) {
            return [];
        }
        const segCount = Math.max(16, segments);
        const pts = [];
        for (let i = 0; i < segCount; i++) {
            const angle = (i / segCount) * Math.PI * 2;
            pts.push({
                x: center.x + Math.cos(angle) * radius,
                y: center.y + Math.sin(angle) * radius,
            });
        }
        return pts;
    }

    function buildCircleHandles(center, radius) {
        if (!center) {
            return [];
        }
        const r = Math.max(radius, 4);
        return [
            { x: center.x + r, y: center.y, index: 0 },
            { x: center.x, y: center.y + r, index: 1 },
            { x: center.x - r, y: center.y, index: 2 },
            { x: center.x, y: center.y - r, index: 3 },
        ];
    }

    function inpaintImageData(imageData, maskData, options = {}) {
        if (!imageData || !maskData) {
            return;
        }
        const width = imageData.width;
        const height = imageData.height;
        const totalPixels = width * height;
        if (!width || !height || totalPixels === 0) {
            return;
        }

        const config = typeof options === "object" && options !== null ? options : {};
        const featherPasses = clamp(Math.round(config.featherPasses ?? 2), 0, 5);
        const diffusionPasses = clamp(Math.round(config.diffusionPasses ?? 8), 0, 20);

        const mask = new Uint8Array(totalPixels);
        const maskBuffer = maskData.data;
        let hasMaskedPixels = false;
        for (let i = 0; i < totalPixels; i++) {
            const masked = maskBuffer[i * 4 + 3] > 10;
            mask[i] = masked ? 1 : 0;
            hasMaskedPixels = hasMaskedPixels || masked;
        }
        if (!hasMaskedPixels) {
            return;
        }

        const source = imageData.data;
        const output = new Uint8ClampedArray(source);
        const dist = new Float32Array(totalPixels);
        const sumR = new Float32Array(totalPixels);
        const sumG = new Float32Array(totalPixels);
        const sumB = new Float32Array(totalPixels);
        const sumW = new Float32Array(totalPixels);
        const INF = 1e9;

        for (let idx = 0; idx < totalPixels; idx++) {
            const offset = idx * 4;
            if (!mask[idx]) {
                dist[idx] = 0;
                sumR[idx] = source[offset];
                sumG[idx] = source[offset + 1];
                sumB[idx] = source[offset + 2];
                sumW[idx] = 1;
            } else {
                dist[idx] = INF;
                sumR[idx] = 0;
                sumG[idx] = 0;
                sumB[idx] = 0;
                sumW[idx] = 0;
            }
        }

        const diagCost = 1.41421356237;
        const forwardNeighbors = [
            { dx: -1, dy: 0, cost: 1 },
            { dx: 0, dy: -1, cost: 1 },
            { dx: -1, dy: -1, cost: diagCost },
            { dx: 1, dy: -1, cost: diagCost },
        ];
        const backwardNeighbors = [
            { dx: 1, dy: 0, cost: 1 },
            { dx: 0, dy: 1, cost: 1 },
            { dx: 1, dy: 1, cost: diagCost },
            { dx: -1, dy: 1, cost: diagCost },
        ];

        function relax(idx, nIdx, cost) {
            const cand = dist[nIdx] + cost;
            const tolerance = 1e-3;
            if (cand + tolerance < dist[idx]) {
                dist[idx] = cand;
                sumR[idx] = sumR[nIdx];
                sumG[idx] = sumG[nIdx];
                sumB[idx] = sumB[nIdx];
                sumW[idx] = sumW[nIdx];
            } else if (Math.abs(cand - dist[idx]) <= tolerance) {
                sumR[idx] += sumR[nIdx];
                sumG[idx] += sumG[nIdx];
                sumB[idx] += sumB[nIdx];
                sumW[idx] += sumW[nIdx];
            }
        }

        // Two full passes (forward + backward) to approximate an 8-connected distance field carrying color.
        for (let pass = 0; pass < 2; pass++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    forwardNeighbors.forEach(({ dx, dy, cost }) => {
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                            return;
                        }
                        const nIdx = ny * width + nx;
                        if (dist[nIdx] + cost < dist[idx]) {
                            relax(idx, nIdx, cost);
                        }
                    });
                }
            }
            for (let y = height - 1; y >= 0; y--) {
                for (let x = width - 1; x >= 0; x--) {
                    const idx = y * width + x;
                    backwardNeighbors.forEach(({ dx, dy, cost }) => {
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                            return;
                        }
                        const nIdx = ny * width + nx;
                        if (dist[nIdx] + cost < dist[idx]) {
                            relax(idx, nIdx, cost);
                        }
                    });
                }
            }
        }

        for (let idx = 0; idx < totalPixels; idx++) {
            if (!mask[idx]) {
                continue;
            }
            const offset = idx * 4;
            const w = sumW[idx] > 0 ? sumW[idx] : 1;
            output[offset] = sumR[idx] / w;
            output[offset + 1] = sumG[idx] / w;
            output[offset + 2] = sumB[idx] / w;
            output[offset + 3] = 255;
        }

        const neighborDeltas = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ];

        if (featherPasses > 0) {
            const temp = new Uint8ClampedArray(output.length);
            for (let pass = 0; pass < featherPasses; pass++) {
                temp.set(output);
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = y * width + x;
                        if (!mask[idx]) {
                            continue;
                        }
                        let r = 0;
                        let g = 0;
                        let b = 0;
                        let count = 0;
                        for (let i = 0; i < neighborDeltas.length; i++) {
                            const dx = neighborDeltas[i][0];
                            const dy = neighborDeltas[i][1];
                            const nx = x + dx;
                            const ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                                continue;
                            }
                            const nIdx = ny * width + nx;
                            const offset = nIdx * 4;
                            r += temp[offset];
                            g += temp[offset + 1];
                            b += temp[offset + 2];
                            count++;
                        }
                        if (count > 0) {
                            const offset = idx * 4;
                            output[offset] = r / count;
                            output[offset + 1] = g / count;
                            output[offset + 2] = b / count;
                            output[offset + 3] = 255;
                        }
                    }
                }
            }
        }

        if (diffusionPasses > 0) {
            const temp = new Uint8ClampedArray(output.length);
            for (let pass = 0; pass < diffusionPasses; pass++) {
                temp.set(output);
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = y * width + x;
                        if (!mask[idx]) {
                            continue;
                        }
                        let r = 0;
                        let g = 0;
                        let b = 0;
                        let count = 0;
                        for (let i = 0; i < neighborDeltas.length; i++) {
                            const dx = neighborDeltas[i][0];
                            const dy = neighborDeltas[i][1];
                            const nx = x + dx;
                            const ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
                                continue;
                            }
                            const nIdx = ny * width + nx;
                            const offset = nIdx * 4;
                            r += temp[offset];
                            g += temp[offset + 1];
                            b += temp[offset + 2];
                            count++;
                        }
                        if (count > 0) {
                            const offset = idx * 4;
                            output[offset] = r / count;
                            output[offset + 1] = g / count;
                            output[offset + 2] = b / count;
                            output[offset + 3] = 255;
                        }
                    }
                }
            }
        }
        imageData.data.set(output);
    }

    function getPolygonDisplayPoints(layer) {
        if (!layer) {
            return [];
        }
        if (!layer.polygonClosed && Array.isArray(layer.tempPolygon) && layer.tempPolygon.length > 0) {
            return layer.tempPolygon;
        }
        return Array.isArray(layer.polygon) ? layer.polygon : [];
    }

    function getPolygonHandlePoints(layer, displayPoints = []) {
        if (!layer) {
            return [];
        }
        if (layer.shapeType === "circle" && layer.shapeMeta?.center) {
            return buildCircleHandles(layer.shapeMeta.center, layer.shapeMeta.radius || 0);
        }
        if (layer.shapeType === "rectangle" && layer.polygonClosed && displayPoints.length >= 4) {
            const handles = displayPoints.slice(0, 4).map((pt, index) => ({
                x: pt.x,
                y: pt.y,
                index,
                role: "corner",
            }));
            for (let i = 0; i < 4; i++) {
                const start = displayPoints[i];
                const end = displayPoints[(i + 1) % 4];
                handles.push({
                    x: (start.x + end.x) / 2,
                    y: (start.y + end.y) / 2,
                    index: `edge-${i}`,
                    role: "edge",
                    edgeIndex: i,
                });
            }
            return handles;
        }
        return displayPoints.map((pt, index) => ({ x: pt.x, y: pt.y, index }));
    }

    function getPolygonHandleAtPoint(layer, cursor, radius = 10) {
        const handles = getPolygonHandlePoints(layer, getPolygonDisplayPoints(layer));
        for (const handle of handles) {
            if (distance(handle, cursor) <= radius) {
                return handle;
            }
        }
        return null;
    }

    function updateRectangleHandleDrag(layer, handleIndex, point, dragContext) {
        if (!layer || layer.polygon.length < 4 || typeof point?.x !== "number" || typeof point?.y !== "number") {
            return;
        }
        const bounds = layer.shapeMeta?.bounds || rectangleBoundsFromPoints(layer.polygon);
        let minX = bounds.minX;
        let minY = bounds.minY;
        let maxX = bounds.maxX;
        let maxY = bounds.maxY;
        const minSize = 1;
        const role = dragContext?.handleRole || "corner";
        if (role === "edge") {
            const edge = dragContext?.edgeIndex ?? -1;
            if (edge === 0) {
                minY = Math.min(point.y, maxY - minSize);
            } else if (edge === 1) {
                maxX = Math.max(point.x, minX + minSize);
            } else if (edge === 2) {
                maxY = Math.max(point.y, minY + minSize);
            } else if (edge === 3) {
                minX = Math.min(point.x, maxX - minSize);
            }
        } else if (typeof handleIndex === "number") {
            switch (handleIndex) {
                case 0:
                    minX = Math.min(point.x, maxX - minSize);
                    minY = Math.min(point.y, maxY - minSize);
                    break;
                case 1:
                    maxX = Math.max(point.x, minX + minSize);
                    minY = Math.min(point.y, maxY - minSize);
                    break;
                case 2:
                    maxX = Math.max(point.x, minX + minSize);
                    maxY = Math.max(point.y, minY + minSize);
                    break;
                case 3:
                    minX = Math.min(point.x, maxX - minSize);
                    maxY = Math.max(point.y, minY + minSize);
                    break;
                default:
                    break;
            }
        }
        maxX = Math.max(maxX, minX + minSize);
        maxY = Math.max(maxY, minY + minSize);
        minX = Math.min(minX, maxX - minSize);
        minY = Math.min(minY, maxY - minSize);
        const rectPoints = buildRectanglePoints({ x: minX, y: minY }, { x: maxX, y: maxY });
        layer.polygon = rectPoints;
        layer.shapeMeta = {
            type: "rectangle",
            bounds: { minX, minY, maxX, maxY },
        };
    }

    function updateCircleHandleDrag(layer, point, dragContext) {
        if (!layer) {
            return;
        }
        const center = dragContext?.circleCenter || layer.shapeMeta?.center;
        if (!center) {
            return;
        }
        const radius = Math.max(distance(center, point), 4);
        layer.shapeMeta = {
            type: "circle",
            center: { x: center.x, y: center.y },
            radius,
        };
        layer.polygon = buildCirclePolygon(center, radius);
    }

    function cloneShapeMeta(meta) {
        if (!meta) {
            return null;
        }
        if (meta.type === "rectangle") {
            return {
                type: "rectangle",
                bounds: meta.bounds
                    ? { minX: meta.bounds.minX, minY: meta.bounds.minY, maxX: meta.bounds.maxX, maxY: meta.bounds.maxY }
                    : null,
            };
        }
        if (meta.type === "circle") {
            return {
                type: "circle",
                center: meta.center ? { x: meta.center.x, y: meta.center.y } : null,
                radius: meta.radius,
            };
        }
        return null;
    }

    function translateShapeMeta(meta, dx, dy) {
        if (!meta) {
            return null;
        }
        if (meta.type === "rectangle" && meta.bounds) {
            return {
                type: "rectangle",
                bounds: {
                    minX: meta.bounds.minX + dx,
                    minY: meta.bounds.minY + dy,
                    maxX: meta.bounds.maxX + dx,
                    maxY: meta.bounds.maxY + dy,
                },
            };
        }
        if (meta.type === "circle" && meta.center) {
            return {
                type: "circle",
                center: { x: meta.center.x + dx, y: meta.center.y + dy },
                radius: meta.radius,
            };
        }
        return cloneShapeMeta(meta);
    }

    function canvasPointFromEvent(evt) {
        const rect = dom.canvas.getBoundingClientRect();
        const x = ((evt.clientX - rect.left) / rect.width) * dom.canvas.width;
        const y = ((evt.clientY - rect.top) / rect.height) * dom.canvas.height;
        return { x, y };
    }

    function notifyParentHeight() {
        if (typeof window === "undefined" || !window.parent) {
            return;
        }
        const height =
            (document.querySelector(".app-shell")?.offsetHeight || document.body?.scrollHeight || dom.canvas?.height || 0) + 40;
        window.parent.postMessage({ type: "WAN2GP_MOTION_DESIGNER_RESIZE", height }, "*");
    }

    function setupHeightObserver() {
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(() => notifyParentHeight());
            const shell = document.querySelector(".app-shell") || document.body;
            observer.observe(shell);    
        } else {
            window.addEventListener("load", notifyParentHeight);
        }
        window.addEventListener("resize", notifyParentHeight);
        notifyParentHeight();
    }

    function findHandle(points, cursor, radius) {
        for (let i = 0; i < points.length; i++) {
            if (distance(points[i], cursor) <= radius) {
                return i;
            }
        }
        return -1;
    }

    function findPathHandleAtPoint(point, radius = 12, preferredLayerId = null) {
        if (preferredLayerId) {
            const preferredLayer = getLayerById(preferredLayerId);
            if (
                preferredLayer &&
                preferredLayer.polygonClosed &&
                Array.isArray(preferredLayer.path) &&
                preferredLayer.path.length > 0
            ) {
                const preferredIndex = findHandle(preferredLayer.path, point, radius);
                if (preferredIndex !== -1) {
                    return { layerId: preferredLayer.id, index: preferredIndex };
                }
            }
        }
        for (let i = state.layers.length - 1; i >= 0; i--) {
            const layer = state.layers[i];
            if (
                !layer ||
                layer.id === preferredLayerId ||
                !layer.polygonClosed ||
                !Array.isArray(layer.path) ||
                layer.path.length === 0
            ) {
                continue;
            }
            const index = findHandle(layer.path, point, radius);
            if (index !== -1) {
                return { layerId: layer.id, index };
            }
        }
        return null;
    }

    function drawPolygonPath(ctx, points) {
        if (points.length === 0) {
            return;
        }
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
    }

    function drawLocalPolygon(ctx, points) {
        if (points.length === 0) {
            return;
        }
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
    }

    function pickLayerFromPoint(point) {
        for (let i = state.layers.length - 1; i >= 0; i--) {
            const layer = state.layers[i];
            if (!layer) {
                continue;
            }
            if (layer.polygon.length >= 3 && pointInPolygon(point, layer.polygon)) {
                return layer;
            }
            if (isPointNearPolygon(point, layer)) {
                return layer;
            }
            if (layer.path && layer.path.length > 0 && isPointNearPath(point, layer)) {
                return layer;
            }
        }
        return null;
    }

    function isPointNearPolygon(point, layer, tolerance = 10) {
        if (!layer || layer.polygon.length === 0) {
            return false;
        }
        for (let i = 0; i < layer.polygon.length; i++) {
            if (distance(point, layer.polygon[i]) <= tolerance) {
                return true;
            }
        }
        const segmentCount = layer.polygonClosed ? layer.polygon.length : layer.polygon.length - 1;
        for (let i = 0; i < segmentCount; i++) {
            const start = layer.polygon[i];
            const end = layer.polygon[(i + 1) % layer.polygon.length];
            if (pointToSegmentDistance(point, start, end) <= tolerance) {
                return true;
            }
        }
        return false;
    }

    function isPointNearPath(point, layer, tolerance = 10) {
        const pathPoints = getRenderPath(layer);
        if (!pathPoints || pathPoints.length === 0) {
            return false;
        }
        for (let i = 0; i < pathPoints.length; i++) {
            if (distance(point, pathPoints[i]) <= tolerance) {
                return true;
            }
        }
        for (let i = 0; i < pathPoints.length - 1; i++) {
            if (pointToSegmentDistance(point, pathPoints[i], pathPoints[i + 1]) <= tolerance) {
                return true;
            }
        }
        return false;
    }

    function getStartPosition(layer) {
        if (!layer || layer.polygon.length === 0) {
            return null;
        }
        return polygonCentroid(layer.polygon);
    }

    function getEffectivePath(layer) {
        if (!layer) {
            return [];
        }
        const start = getStartPosition(layer);
        const nodes = Array.isArray(layer.path) ? layer.path.slice() : [];
        return start ? [start, ...nodes] : nodes;
    }

    function updateLayerPathCache(layer) {
        if (!layer) {
            return;
        }
        const basePath = getEffectivePath(layer);
        if (!Array.isArray(basePath) || basePath.length === 0) {
            layer.renderPath = [];
            layer.renderSegmentMap = [];
            layer.pathMeta = null;
            return;
        }
        if (basePath.length === 1) {
            layer.renderPath = basePath.slice();
            layer.renderSegmentMap = [];
            layer.pathMeta = computePathMeta(layer.renderPath);
            return;
        }
        const tensionAmount = clamp(layer.tension ?? 0, 0, 1);
        if (tensionAmount > 0) {
            const curved = buildRenderPath(basePath, tensionAmount);
            layer.renderPath = curved.points;
            layer.renderSegmentMap = curved.segmentMap;
        } else {
            layer.renderPath = basePath.slice();
            layer.renderSegmentMap = [];
            for (let i = 0; i < layer.renderPath.length - 1; i++) {
                layer.renderSegmentMap.push(i);
            }
        }
        layer.pathMeta = computePathMeta(layer.renderPath);
    }

    function buildRenderPath(points, tensionAmount) {
        const samples = Math.max(6, Math.round(8 + tensionAmount * 20));
        const cardinalTension = clamp(1 - tensionAmount, 0, 1);
        const rendered = [points[0]];
        const segmentMap = [];
        for (let i = 0; i < points.length - 1; i++) {
            const p0 = points[i - 1] || points[i];
            const p1 = points[i];
            const p2 = points[i + 1];
            const p3 = points[i + 2] || p2;
            for (let step = 1; step <= samples; step++) {
                const t = step / samples;
                rendered.push(cardinalPoint(p0, p1, p2, p3, t, cardinalTension));
                segmentMap.push(i);
            }
        }
        return { points: rendered, segmentMap };
    }

    function cardinalPoint(p0, p1, p2, p3, t, tension) {
        const t2 = t * t;
        const t3 = t2 * t;
        const s = (1 - tension) / 2;
        const m1x = (p2.x - p0.x) * s;
        const m1y = (p2.y - p0.y) * s;
        const m2x = (p3.x - p1.x) * s;
        const m2y = (p3.y - p1.y) * s;
        const h00 = 2 * t3 - 3 * t2 + 1;
        const h10 = t3 - 2 * t2 + t;
        const h01 = -2 * t3 + 3 * t2;
        const h11 = t3 - t2;
        return {
            x: h00 * p1.x + h10 * m1x + h01 * p2.x + h11 * m2x,
            y: h00 * p1.y + h10 * m1y + h01 * p2.y + h11 * m2y,
        };
    }

    function findBaseSegmentIndex(layer, point, tolerance = 10) {
        if (!layer) {
            return -1;
        }
        const basePoints = getEffectivePath(layer);
        if (!basePoints || basePoints.length < 2) {
            return -1;
        }
        for (let i = 0; i < basePoints.length - 1; i++) {
            if (pointToSegmentDistance(point, basePoints[i], basePoints[i + 1]) <= tolerance) {
                return i;
            }
        }
        const renderPoints = getRenderPath(layer);
        const map = Array.isArray(layer.renderSegmentMap) ? layer.renderSegmentMap : [];
        if (renderPoints.length > 1 && map.length === renderPoints.length - 1) {
            for (let i = 0; i < renderPoints.length - 1; i++) {
                if (pointToSegmentDistance(point, renderPoints[i], renderPoints[i + 1]) <= tolerance) {
                    const mapped = map[i];
                    if (typeof mapped === "number" && mapped >= 0) {
                        return mapped;
                    }
                }
            }
        }
        return -1;
    }

    function pointInPolygon(point, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i].x;
            const yi = polygon[i].y;
            const xj = polygon[j].x;
            const yj = polygon[j].y;
            const intersect = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-8) + xi;
            if (intersect) {
                inside = !inside;
            }
        }
        return inside;
    }

    function polygonBounds(points) {
        const xs = points.map((p) => p.x);
        const ys = points.map((p) => p.y);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);
        return {
            minX,
            minY,
            width: maxX - minX || 1,
            height: maxY - minY || 1,
        };
    }

    function polygonCentroid(points) {
        let x = 0;
        let y = 0;
        let signedArea = 0;
        for (let i = 0; i < points.length; i++) {
            const p0 = points[i];
            const p1 = points[(i + 1) % points.length];
            const a = p0.x * p1.y - p1.x * p0.y;
            signedArea += a;
            x += (p0.x + p1.x) * a;
            y += (p0.y + p1.y) * a;
        }
        signedArea *= 0.5;
        if (signedArea === 0) {
            return points[0];
        }
        x = x / (6 * signedArea);
        y = y / (6 * signedArea);
        return { x, y };
    }

    function computePathMeta(points) {
        if (points.length < 2) {
            return null;
        }
        const lengths = [0];
        let total = 0;
        for (let i = 1; i < points.length; i++) {
            total += distance(points[i - 1], points[i]);
            lengths.push(total);
        }
        return { lengths, total };
    }

    function distance(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    function lerp(a, b, t) {
        return a + (b - a) * clamp(t, 0, 1);
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function insertPointOnPolygonEdge(layer, point, tolerance = 10) {
        if (
            !layer ||
            (layer.shapeType && layer.shapeType !== "polygon") ||
            !Array.isArray(layer.polygon) ||
            layer.polygon.length < 2
        ) {
            return false;
        }
        const segments = layer.polygonClosed ? layer.polygon.length : layer.polygon.length - 1;
        for (let i = 0; i < segments; i++) {
            const start = layer.polygon[i];
            const end = layer.polygon[(i + 1) % layer.polygon.length];
            if (pointToSegmentDistance(point, start, end) <= tolerance) {
                layer.polygon.splice(i + 1, 0, { x: point.x, y: point.y });
                layer.selectedPolygonIndex = i + 1;
                if (layer.polygonClosed) {
                    computeLayerAssets(layer);
                }
                updateBadge();
                updateActionAvailability();
                return true;
            }
        }
        return false;
    }

    function insertPointOnPathEdge(layer, point, tolerance = 10) {
        if (!layer || !layer.polygonClosed || !Array.isArray(layer.path)) {
            return false;
        }
        const segmentIndex = findBaseSegmentIndex(layer, point, tolerance);
        if (segmentIndex === -1) {
            return false;
        }
        const insertIndex = Math.max(segmentIndex, 0);
        layer.path.splice(insertIndex, 0, { x: point.x, y: point.y });
        layer.selectedPathIndex = insertIndex;
        updateLayerPathCache(layer);
        updateBadge();
        updateActionAvailability();
        return true;
    }

    function removePolygonPoint(layer, point, tolerance = 8) {
        if (!layer || (layer.shapeType && layer.shapeType !== "polygon") || layer.polygon.length === 0) {
            return false;
        }
        for (let i = 0; i < layer.polygon.length; i++) {
            if (distance(point, layer.polygon[i]) <= tolerance) {
                layer.polygon.splice(i, 1);
                layer.selectedPolygonIndex = layer.polygon.length > 0 ? Math.min(i, layer.polygon.length - 1) : -1;
                if (layer.polygon.length < 3) {
                    layer.polygonClosed = false;
                    layer.objectCut = null;
                    recomputeBackgroundFill();
                } else if (layer.polygonClosed) {
                    computeLayerAssets(layer);
                }
                updateBadge();
                updateActionAvailability();
                updateLayerPathCache(layer);
                return true;
            }
        }
        return false;
    }

    function removePathPoint(layer, point, tolerance = 8) {
        if (!layer || !Array.isArray(layer.path) || layer.path.length === 0) {
            return false;
        }
        for (let i = 0; i < layer.path.length; i++) {
            if (distance(point, layer.path[i]) <= tolerance) {
                layer.path.splice(i, 1);
                layer.selectedPathIndex = layer.path.length > 0 ? Math.min(i, layer.path.length - 1) : -1;
                if (layer.path.length <= 1) {
                    layer.pathLocked = false;
                }
                updateLayerPathCache(layer);
                updateBadge();
                updateActionAvailability();
                return true;
            }
        }
        return false;
    }

    function pointToSegmentDistance(point, start, end) {
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lengthSq = dx * dx + dy * dy;
        if (lengthSq === 0) {
            return distance(point, start);
        }
        let t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSq;
        t = Math.max(0, Math.min(1, t));
        const proj = { x: start.x + t * dx, y: start.y + t * dy };
        return distance(point, proj);
    }
})();
