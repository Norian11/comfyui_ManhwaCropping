import { app } from "../../scripts/app.js";

const CROP_NODE = "Manhwa Crop";
const STITCH_NODE = "Manhwa Stitch Save";
const PAD = 10;
const MIN_W = 320;
const MIN_H = 460;

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function hideWidget(node, name) {
    const w = getWidget(node, name);
    if (!w || w.__hidden_manhwa) return;
    w.__hidden_manhwa = true;
    w.computeSize = () => [0, -4];
    w.type = "converted-widget";
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function computeCanvasSize(node, nodeSize) {
    if (!node.widgets || !node.widgets.length) return;
    const firstWidgetY = node.widgets[0].last_y || 0;
    node.canvasHeight = Math.max(240, nodeSize[1] - firstWidgetY - 20);
}

function getInputImageURL(node) {
    const w = getWidget(node, "image");
    if (!w || !w.value) return null;
    const filename = String(w.value).trim();
    if (!filename) return null;
    return `/view?filename=${encodeURIComponent(filename)}&type=input`;
}

function toImageCoords(state, px, py) {
    const p = state.previewRect;
    const img = state.image;
    return {
        x: Math.floor((px - p.x) * (img.width / p.w)),
        y: Math.floor((py - p.y) * (img.height / p.h)),
    };
}

function updateCropWidgets(node, x, y, size) {
    const xw = getWidget(node, "x");
    const yw = getWidget(node, "y");
    const sw = getWidget(node, "size");
    if (xw) xw.value = x;
    if (yw) yw.value = y;
    if (sw) sw.value = size;
    node.setDirtyCanvas(true, true);
}

function snapToClosestMultiple16(value, minValue, maxValue) {
    if (maxValue < minValue) return minValue;
    const snapped = Math.round(value / 16) * 16;
    return clamp(snapped, minValue, maxValue);
}

function snapSelectionTo16(node, state) {
    if (!state.image) return;
    const img = state.image;
    const xw = getWidget(node, "x");
    const yw = getWidget(node, "y");
    const sw = getWidget(node, "size");
    if (!xw || !yw || !sw) return;

    let x = Number(xw.value) || 0;
    let y = Number(yw.value) || 0;
    let size = Number(sw.value) || 16;
    size = snapToClosestMultiple16(size, 16, 100000);
    x = Math.round(x);
    y = Math.round(y);
    updateCropWidgets(node, x, y, size);
}

function drawCropOverlay(node, cctx, state) {
    if (!state.previewRect || !state.image) return;
    const p = state.previewRect;
    const img = state.image;
    const xw = getWidget(node, "x");
    const yw = getWidget(node, "y");
    const sw = getWidget(node, "size");
    const ix = xw ? Number(xw.value) || 0 : 0;
    const iy = yw ? Number(yw.value) || 0 : 0;
    const isz = sw ? Number(sw.value) || 1 : 1;

    const x = p.x + (ix / img.width) * p.w;
    const y = p.y + (iy / img.height) * p.h;
    const s = (isz / img.width) * p.w;

    cctx.strokeStyle = "#00d4ff";
    cctx.lineWidth = 2;
    cctx.strokeRect(x, y, s, s);
    cctx.fillStyle = "rgba(0,212,255,0.12)";
    cctx.fillRect(x, y, s, s);
}

function drawPreview(node, ctx, widgetWidth, widgetY) {
    const state = node.__manhwa;
    if (!state) return;

    const t = ctx.getTransform();
    const h = Math.max(240, node.canvasHeight || 320);

    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, Math.floor(widgetWidth * t.a));
    const cssH = Math.max(1, Math.floor(h * t.d));

    const img = state.image;
    if (!img || !img.width || !img.height) {
        state.previewRect = null;
        state.canvas.hidden = true;
        return;
    }

    const availW = Math.max(1, cssW - PAD * 2);
    const availH = Math.max(1, cssH - PAD * 2);
    const scale = Math.min(availW / img.width, availH / img.height);
    const drawW = img.width * scale;
    const drawH = img.height * scale;
    const drawX = PAD + (availW - drawW) * 0.5;
    const drawY = PAD + (availH - drawH) * 0.5;
    state.previewRect = { x: 0, y: 0, w: drawW, h: drawH };

    Object.assign(state.canvas.style, {
        left: `${t.e + drawX}px`,
        top: `${t.f + (widgetY * t.d) + drawY}px`,
        width: `${drawW}px`,
        height: `${drawH}px`,
        position: "absolute",
        zIndex: 5,
        pointerEvents: "auto",
    });
    state.canvas.hidden = false;
    state.canvas.width = Math.max(1, Math.floor(drawW * dpr));
    state.canvas.height = Math.max(1, Math.floor(drawH * dpr));

    const cctx = state.canvas.getContext("2d");
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.clearRect(0, 0, drawW, drawH);
    cctx.drawImage(img, 0, 0, drawW, drawH);
    cctx.strokeStyle = "#3a3a3a";
    cctx.strokeRect(0, 0, drawW, drawH);
    drawCropOverlay(node, cctx, state);
}

function installCropPointerHandlers(node) {
    const state = node.__manhwa;
    const canvas = state.canvas;

    const getLocal = (evt) => {
        const r = canvas.getBoundingClientRect();
        return { x: evt.clientX - r.left, y: evt.clientY - r.top };
    };

    const inPreview = (pt) => {
        const p = state.previewRect;
        if (!p) return false;
        return pt.x >= p.x && pt.x <= p.x + p.w && pt.y >= p.y && pt.y <= p.y + p.h;
    };

    const onMove = (evt) => {
        if (!state.dragging || !state.start || !state.previewRect || !state.image) return;
        const pt = getLocal(evt);
        const cur = toImageCoords(state, pt.x, pt.y);
        const start = state.startImage;

        const dx = cur.x - start.x;
        const dy = cur.y - start.y;
        let side = Math.max(Math.abs(dx), Math.abs(dy));
        side = clamp(side, 1, 100000);

        let x = dx >= 0 ? start.x : (start.x - side);
        let y = dy >= 0 ? start.y : (start.y - side);

        updateCropWidgets(node, x, y, side);
    };

    const endDrag = () => {
        if (!state.dragging) return;
        state.dragging = false;
        snapSelectionTo16(node, state);
        window.removeEventListener("mousemove", onMove, true);
        window.removeEventListener("mouseup", endDrag, true);
    };

    canvas.addEventListener("mousedown", (evt) => {
        if (evt.button !== 0 || !state.previewRect || !state.image) return;
        const pt = getLocal(evt);
        if (!inPreview(pt)) return;
        evt.preventDefault();
        evt.stopPropagation();
        state.dragging = true;
        state.start = { x: pt.x, y: pt.y };
        state.startImage = toImageCoords(state, pt.x, pt.y);
        window.addEventListener("mousemove", onMove, true);
        window.addEventListener("mouseup", endDrag, true);
    });
}

function attachCropCanvasWidget(node) {
    const widget = {
        type: "customCanvas",
        name: "manhwa-crop-canvas",
        draw: function (ctx, n, width, y) {
            drawPreview(node, ctx, width, y);
        },
    };
    const canvas = document.createElement("canvas");
    canvas.className = "manhwa-crop-canvas";
    widget.canvas = canvas;
    document.body.appendChild(canvas);
    node.addCustomWidget(widget);
    node.__manhwa.canvas = canvas;
    installCropPointerHandlers(node);
}

function setupStitchButton(node) {
    hideWidget(node, "stitch_and_save_trigger");
    const trigger = getWidget(node, "stitch_and_save_trigger");
    if (!trigger || node.__manhwa_stitch_btn) return;

    node.__manhwa_stitch_btn = node.addWidget("button", "Stitch and Save", null, async () => {
        trigger.value = (Number(trigger.value) || 0) + 1;
        node.setDirtyCanvas(true, true);
        try {
            await app.queuePrompt(0, 1);
        } catch (e) {
            app.queuePrompt(0, 1);
        }
    });
}

app.registerExtension({
    name: "comfyui.manhwa.crop",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === CROP_NODE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const out = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.__manhwa = {
                    canvas: null,
                    image: null,
                    imageSrc: null,
                    previewRect: null,
                    dragging: false,
                    start: null,
                };
                this.size = this.size || [MIN_W, MIN_H];
                this.size[0] = Math.max(this.size[0], MIN_W);
                this.size[1] = Math.max(this.size[1], MIN_H);
                hideWidget(this, "x");
                hideWidget(this, "y");
                hideWidget(this, "size");
                computeCanvasSize(this, this.size);
                attachCropCanvasWidget(this);
                return out;
            };

            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function (size) {
                if (onResize) onResize.apply(this, arguments);
                computeCanvasSize(this, size);
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (this.__manhwa && this.__manhwa.canvas && this.__manhwa.canvas.parentNode) {
                    this.__manhwa.canvas.parentNode.removeChild(this.__manhwa.canvas);
                }
                if (onRemoved) onRemoved.apply(this, arguments);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function () {
                if (onExecuted) onExecuted.apply(this, arguments);
                this.setDirtyCanvas(true, true);
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);
                const st = this.__manhwa;
                if (!st) return;
                const src = getInputImageURL(this);
                if (src && src !== st.imageSrc) {
                    st.imageSrc = src;
                    const img = new Image();
                    img.onload = () => {
                        st.image = img;
                        this.setDirtyCanvas(true, true);
                    };
                    img.src = src;
                }
            };
        }

        if (nodeData.name === STITCH_NODE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const out = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                setupStitchButton(this);
                return out;
            };
        }

        if (!app.__manhwa_drawbg_patched) {
            app.__manhwa_drawbg_patched = true;
            const originalDrawBg = app.canvas.onDrawBackground;
            app.canvas.onDrawBackground = function () {
                if (originalDrawBg) originalDrawBg.apply(this, arguments);
                for (const n of app.graph._nodes) {
                    if (!n.widgets) continue;
                    for (const w of n.widgets) {
                        if (w.canvas) {
                            w.canvas.style.left = "-10000px";
                            w.canvas.style.top = "-10000px";
                        }
                    }
                }
            };
        }
    },
});
