(() => {
    "use strict";

    const STORAGE_KEY = "echo-blog-site-zoom";
    const DEFAULT_ZOOM = 1;
    const MIN_ZOOM = 0.6;
    const MAX_ZOOM = 2;
    const STEP = 0.1;
    let hideTimer;

    const clamp = (value) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));

    const readStoredZoom = () => {
        try {
            const value = Number.parseFloat(localStorage.getItem(STORAGE_KEY));
            return Number.isFinite(value) ? clamp(value) : DEFAULT_ZOOM;
        } catch {
            return DEFAULT_ZOOM;
        }
    };

    let currentZoom = readStoredZoom();

    const applyZoom = () => {
        document.documentElement.style.zoom = String(currentZoom);
    };

    const saveZoom = () => {
        try {
            localStorage.setItem(STORAGE_KEY, String(currentZoom));
        } catch {
            // The zoom still works when storage is unavailable.
        }
    };

    const getIndicator = () => {
        let indicator = document.querySelector(".site-zoom-indicator");
        if (indicator) {
            return indicator;
        }

        indicator = document.createElement("div");
        indicator.className = "site-zoom-indicator";
        indicator.setAttribute("role", "status");
        indicator.setAttribute("aria-live", "polite");
        document.body.appendChild(indicator);
        return indicator;
    };

    const showIndicator = () => {
        const indicator = getIndicator();
        indicator.textContent = `${Math.round(currentZoom * 100)}%`;
        indicator.classList.add("is-visible");
        window.clearTimeout(hideTimer);
        hideTimer = window.setTimeout(() => {
            indicator.classList.remove("is-visible");
        }, 900);
    };

    const setZoom = (value) => {
        currentZoom = Math.round(clamp(value) * 10) / 10;
        applyZoom();
        saveZoom();
        showIndicator();
    };

    applyZoom();

    window.addEventListener(
        "wheel",
        (event) => {
            if (!event.ctrlKey && !event.metaKey) {
                return;
            }

            event.preventDefault();
            setZoom(currentZoom + (event.deltaY < 0 ? STEP : -STEP));
        },
        { passive: false },
    );

    window.addEventListener("keydown", (event) => {
        if ((!event.ctrlKey && !event.metaKey) || event.key !== "0") {
            return;
        }

        event.preventDefault();
        setZoom(DEFAULT_ZOOM);
    });
})();
