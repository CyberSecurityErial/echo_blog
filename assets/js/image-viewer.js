(() => {
    "use strict";

    const images = [...document.querySelectorAll(".post-content img")]
        .filter((image) => !image.closest("a"));
    if (!images.length || typeof HTMLDialogElement === "undefined") return;

    const dialog = document.createElement("dialog");
    dialog.className = "image-viewer";
    dialog.setAttribute("aria-label", "图片查看器");
    dialog.setAttribute("aria-describedby", "image-viewer-hint");
    dialog.innerHTML = `
        <div class="image-viewer-toolbar">
            <button type="button" data-action="out" aria-label="缩小图片">−</button>
            <output aria-label="缩放比例"></output>
            <button type="button" data-action="in" aria-label="放大图片">+</button>
            <button type="button" data-action="fit">适应窗口</button>
            <button type="button" data-action="actual">原始大小</button>
            <button type="button" data-action="close" autofocus>关闭</button>
        </div>
        <p class="image-viewer-hint" id="image-viewer-hint">滚轮缩放 · 拖动查看 · Esc 关闭</p>
        <div class="image-viewer-stage"><img alt="" draggable="false"></div>`;
    document.body.appendChild(dialog);

    const stage = dialog.querySelector(".image-viewer-stage");
    const preview = stage.querySelector("img");
    const percentage = dialog.querySelector("output");
    let width = 0;
    let height = 0;
    let scale = 1;
    let x = 0;
    let y = 0;
    let source;
    let drag;

    const render = () => {
        preview.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
        percentage.value = `${Math.round(scale * 100)}%`;
    };

    const center = () => {
        x = (stage.clientWidth - width * scale) / 2;
        y = (stage.clientHeight - height * scale) / 2;
        render();
    };

    const fit = () => {
        if (!width || !height) return;
        scale = Math.min(1, (stage.clientWidth - 24) / width, (stage.clientHeight - 24) / height);
        center();
    };

    const zoom = (next, atX = stage.clientWidth / 2, atY = stage.clientHeight / 2) => {
        if (!width || !height) return;
        const minimum = Math.min(0.1, (stage.clientWidth - 24) / width, (stage.clientHeight - 24) / height);
        next = Math.max(minimum, Math.min(8, next));
        const ratio = next / scale;
        x = atX - (atX - x) * ratio;
        y = atY - (atY - y) * ratio;
        scale = next;
        render();
    };

    const open = (image) => {
        if (!image.naturalWidth || !image.naturalHeight) return;
        source = image;
        width = image.naturalWidth;
        height = image.naturalHeight;
        preview.src = image.currentSrc || image.src;
        preview.alt = image.alt;
        preview.style.width = `${width}px`;
        preview.style.height = `${height}px`;
        // Keep the viewer in screen coordinates when the page has saved CSS zoom.
        const pageZoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
        dialog.style.zoom = String(1 / pageZoom);
        document.documentElement.classList.add("image-viewer-open");
        dialog.showModal();
        fit();
    };

    for (const image of images) {
        image.dataset.zoomable = "";
        image.tabIndex = 0;
        image.setAttribute("role", "button");
        image.setAttribute("aria-haspopup", "dialog");
        image.setAttribute("aria-label", `放大查看：${image.alt || "图片"}`);
        image.title = image.title || "点击放大，滚轮缩放";
        image.addEventListener("click", () => open(image));
        image.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                open(image);
            }
        });
    }

    dialog.addEventListener("click", (event) => {
        const action = event.target.closest("button")?.dataset.action;
        if (action === "in") zoom(scale * 1.25);
        if (action === "out") zoom(scale / 1.25);
        if (action === "fit") fit();
        if (action === "actual") { scale = 1; center(); }
        if (action === "close") dialog.close();
    });

    dialog.addEventListener("wheel", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!stage.contains(event.target)) return;
        const rect = stage.getBoundingClientRect();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? stage.clientHeight : 1;
        const delta = Math.max(-100, Math.min(100, event.deltaY * unit));
        zoom(scale * Math.exp(-delta * 0.002), event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    dialog.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (["+", "=", "-", "0"].includes(event.key)) {
            event.preventDefault();
            if (event.key === "0") fit();
            else zoom(scale * (event.key === "-" ? 0.8 : 1.25));
        }
    });

    stage.addEventListener("pointerdown", (event) => {
        if (event.button !== 0 || drag) return;
        drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
        stage.setPointerCapture(event.pointerId);
        stage.classList.add("is-dragging");
    });
    stage.addEventListener("pointermove", (event) => {
        if (!drag || drag.id !== event.pointerId) return;
        x += event.clientX - drag.x;
        y += event.clientY - drag.y;
        drag.x = event.clientX;
        drag.y = event.clientY;
        render();
    });
    const endDrag = () => {
        drag = null;
        stage.classList.remove("is-dragging");
    };
    stage.addEventListener("lostpointercapture", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    dialog.addEventListener("close", () => {
        endDrag();
        document.documentElement.classList.remove("image-viewer-open");
        source?.focus({ preventScroll: true });
    });
    window.addEventListener("resize", () => { if (dialog.open) fit(); });
})();
