(() => {
    "use strict";

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const metricSeeds = [
        { label: "NULL SCORE", value: 184, min: 90, max: 260, unit: "" },
        { label: "MOOD BUS", value: 37, min: 12, max: 96, unit: "%" },
        { label: "SALT INDEX", value: 4096, min: 2048, max: 8192, unit: "" },
        { label: "CACHE OMEN", value: 13, min: 0, max: 99, unit: "/99" },
        { label: "SLEEP DEBT", value: 26, min: 0, max: 72, unit: "h" },
        { label: "ACK DELAY", value: 8, min: 1, max: 48, unit: "ms" },
    ];
    const logLines = [
        "SYS: sampling ambient entropy",
        "BLOG: indexing unposted thoughts",
        "KVCACHE: folding spare tokens",
        "SCHED: slot drift accepted",
        "NCCL: imaginary rank synchronized",
        "MOODBUS: optimism clamped",
        "SEC: salt rotated into decimal",
        "AGENT: waiting for impossible ACK",
        "TRACE: static pressure nominal",
        "ROUTER: regret path pruned",
        "MEM: yesterday checksum = NaN",
        "CLOCK: local time disagrees politely",
    ];

    const pad = (value, size = 2) => String(value).padStart(size, "0");
    const randomStep = () => Math.floor(Math.random() * 9) - 4;
    const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

    const createMetric = (metric) => {
        const item = document.createElement("div");
        item.className = "tech-terminal-metric";

        const label = document.createElement("span");
        label.textContent = metric.label;

        const value = document.createElement("strong");
        value.dataset.metricValue = metric.label;
        value.textContent = `${metric.value}${metric.unit}`;

        item.append(label, value);
        return item;
    };

    const formatTime = () => {
        const now = new Date();
        return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    };

    const appendLog = (list, index) => {
        const item = document.createElement("li");
        item.textContent = `[${formatTime()}] ${logLines[index % logLines.length]}`;
        list.prepend(item);

        while (list.children.length > 7) {
            list.lastElementChild.remove();
        }
    };

    const buildTerminal = () => {
        const dock = document.createElement("aside");
        dock.className = "tech-terminal-dock";
        dock.setAttribute("aria-label", "terminal telemetry");

        const title = document.createElement("div");
        title.className = "tech-terminal-title";

        const titleText = document.createElement("span");
        titleText.textContent = "ODD TELEMETRY";

        const state = document.createElement("span");
        state.className = "tech-terminal-rec";
        state.textContent = "PRINT";

        title.append(titleText, state);

        const grid = document.createElement("div");
        grid.className = "tech-terminal-grid";
        metricSeeds.forEach((metric) => grid.append(createMetric(metric)));

        const log = document.createElement("ol");
        log.className = "tech-terminal-log";
        log.setAttribute("aria-live", "polite");

        const cursor = document.createElement("div");
        cursor.className = "tech-terminal-cursor";
        cursor.textContent = ">_";

        dock.append(title, grid, log, cursor);
        return { dock, log };
    };

    const updateMetrics = (dock) => {
        metricSeeds.forEach((metric) => {
            metric.value = clamp(metric.value + randomStep(), metric.min, metric.max);
            const value = dock.querySelector(`[data-metric-value="${metric.label}"]`);
            if (value) {
                value.textContent = `${metric.value}${metric.unit}`;
            }
        });
    };

    const boot = () => {
        if (document.querySelector(".tech-terminal-dock")) {
            return;
        }

        const { dock, log } = buildTerminal();
        const main = document.querySelector(".main");
        if (main) {
            main.before(dock);
        } else {
            document.body.appendChild(dock);
        }

        for (let index = 0; index < 4; index += 1) {
            appendLog(log, index);
        }

        let logIndex = 4;
        window.setInterval(() => updateMetrics(dock), prefersReducedMotion ? 2400 : 900);
        window.setInterval(() => {
            appendLog(log, logIndex);
            logIndex += 1;
        }, prefersReducedMotion ? 4200 : 1600);
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        boot();
    }
})();
