/* Neuro Trainer — веб-интерфейс. Форма параметров строится по каталогу (/api/meta). */
(function () {
  "use strict";

  const BASE = document.body.dataset.base || "";
  const POLL_ACTIVE_MS = 2000;
  const POLL_IDLE_MS = 6000;
  const SAVE_DEBOUNCE_MS = 400;
  const TOOLTIP_DELAY_MS = 250;
  const SERIES_COLORS = ["#2f6fde", "#e0782f", "#2e8b57", "#b04fc4", "#c93a3a", "#7a8b99", "#c9a227", "#1aa3a3"];

  const S = {
    meta: null,
    state: {},
    errors: {},
    paramsByKey: {},
    pending: {},
    saveTimer: null,
    job: null,
    rows: [],
    pollTimer: null,
  };

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // ---------------------------------------------------------------- API
  async function api(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(BASE + path, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
      throw new Error(detail);
    }
    return data;
  }

  function showMessage(text, kind) {
    const el = $("#message");
    if (!text) { el.hidden = true; return; }
    el.textContent = text;
    el.className = "message" + (kind ? " " + kind : "");
    el.hidden = false;
  }

  // ---------------------------------------------------------------- вкладки
  function switchTab(name) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    $$(".tab-page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  }

  // ---------------------------------------------------------------- подсказки
  function initTooltip() {
    const tip = $("#tooltip");
    let timer = null;
    let current = null;

    function helpFor(el) {
      if (el.dataset.help) return { text: el.dataset.help, key: el.dataset.paramKey || "" };
      if (el.dataset.helpKey && S.meta) return { text: S.meta.help[el.dataset.helpKey] || "", key: "" };
      return null;
    }

    document.addEventListener("mouseover", (e) => {
      const el = e.target.closest("[data-help], [data-help-key]");
      if (el === current) return;
      current = el;
      clearTimeout(timer);
      tip.hidden = true;
      if (!el) return;
      const h = helpFor(el);
      if (!h || !h.text) return;
      timer = setTimeout(() => {
        tip.innerHTML = "";
        tip.appendChild(document.createTextNode(h.text));
        if (h.key) {
          const k = document.createElement("span");
          k.className = "tip-key";
          k.textContent = "параметр Ultralytics: " + h.key;
          tip.appendChild(k);
        }
        tip.hidden = false;
        const r = el.getBoundingClientRect();
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        let left = Math.min(r.left, window.innerWidth - tw - 8);
        let top = r.bottom + 6;
        if (top + th > window.innerHeight - 8) top = r.top - th - 6;
        tip.style.left = Math.max(8, left) + "px";
        tip.style.top = Math.max(8, top) + "px";
      }, TOOLTIP_DELAY_MS);
    });
    document.addEventListener("scroll", () => { tip.hidden = true; }, true);
  }

  // ---------------------------------------------------------------- форма параметров
  function makeParamField(p) {
    const field = document.createElement("div");
    field.className = "field" + (p.kind === "bool" ? " bool" : "");
    field.dataset.key = p.key;
    field.dataset.help = p.help;
    field.dataset.paramKey = p.key;
    if (p.tasks && p.tasks.length) field.dataset.tasks = p.tasks.join(",");

    const id = "p-" + p.key;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = p.label;

    let input;
    if (p.kind === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
    } else if (p.kind === "choice") {
      input = document.createElement("select");
      p.choices.forEach((c) => {
        const o = document.createElement("option");
        o.value = c; o.textContent = c;
        input.appendChild(o);
      });
    } else if (p.kind === "imgsz") {
      input = document.createElement("input");
      input.type = "text";
      input.placeholder = "640 или 480x640";
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (p.min !== null) input.min = p.min;
      if (p.max !== null) input.max = p.max;
      input.step = p.step !== null ? p.step : (p.kind === "int" ? 1 : "any");
    }
    input.id = id;
    input.addEventListener("change", () => queueSave({ [p.key]: readParam(p, input) }));

    if (p.kind === "bool") {
      field.appendChild(input);
      field.appendChild(label);
    } else {
      field.appendChild(label);
      field.appendChild(input);
    }
    const err = document.createElement("div");
    err.className = "err-text";
    err.hidden = true;
    field.appendChild(err);
    return field;
  }

  function readParam(p, input) {
    if (p.kind === "bool") return input.checked;
    if (p.kind === "int" || p.kind === "float") return input.value === "" ? null : Number(input.value);
    return input.value;
  }

  function writeParam(p, value) {
    const input = document.getElementById("p-" + p.key);
    if (!input || value === undefined || value === null) return;
    if (p.kind === "bool") input.checked = !!value;
    else input.value = String(value);
  }

  function buildParamGroups() {
    const adv = $("#advanced-groups");
    S.meta.groups.forEach((g) => {
      g.params.forEach((p) => { S.paramsByKey[p.key] = p; });
      let container;
      if (g.advanced) {
        const sub = document.createElement("div");
        sub.className = "subgroup";
        sub.dataset.groupKey = g.key;
        const title = document.createElement("div");
        title.className = "subgroup-title";
        title.textContent = g.title;
        container = document.createElement("div");
        container.className = "params";
        sub.appendChild(title);
        sub.appendChild(container);
        adv.appendChild(sub);
      } else {
        container = $(`.params[data-group="${g.key}"]`);
      }
      g.params.forEach((p) => container.appendChild(makeParamField(p)));
    });
  }

  function applyTaskVisibility() {
    const task = S.state.task;
    $$(".field[data-tasks]").forEach((f) => {
      f.hidden = !f.dataset.tasks.split(",").includes(task);
    });
    $$(".subgroup").forEach((sub) => {
      sub.hidden = $$(".field", sub).every((f) => f.hidden);
    });
  }

  // ---------------------------------------------------------------- модель
  function familyByKey(key) { return S.meta.families.find((f) => f.key === key) || S.meta.families[0]; }

  function familyOfModel(name) {
    return S.meta.families.find((f) => Object.values(f.models).some((list) => list.includes(name)));
  }

  function fillSelect(sel, items, value) {
    sel.innerHTML = "";
    items.forEach((it) => {
      const o = document.createElement("option");
      o.value = it.value; o.textContent = it.label;
      sel.appendChild(o);
    });
    if (value !== undefined && items.some((it) => it.value === value)) sel.value = value;
  }

  function sizeLetter(name, family) {
    // yolo26n-seg.pt → "n"
    const rest = name.slice(family.key.length);
    return rest.charAt(0);
  }

  function renderModelSelectors() {
    const st = S.state;
    const fam = familyOfModel(st.model_name) || S.meta.families[0];
    const famSel = $("#f-family");
    fillSelect(famSel, S.meta.families.map((f) => ({ value: f.key, label: f.label })), fam.key);

    const taskItems = S.meta.tasks.filter((t) => fam.models[t.value]).map((t) => ({ value: t.value, label: t.label }));
    fillSelect($("#f-task"), taskItems, st.task);

    const task = $("#f-task").value;
    const models = (fam.models[task] || []).map((m) => ({ value: m, label: m }));
    fillSelect($("#f-model_name"), models, st.model_name);
  }

  function pickModel(familyKey, task, preferSize) {
    const fam = familyByKey(familyKey);
    let t = task;
    if (!fam.models[t]) t = Object.keys(fam.models)[0];
    const list = fam.models[t];
    const same = list.find((m) => sizeLetter(m, fam) === preferSize);
    return same || list[0];
  }

  function onModelSelectorChange(source) {
    const fam = familyByKey($("#f-family").value);
    const current = S.state.model_name;
    const curFam = familyOfModel(current);
    const size = curFam ? sizeLetter(current, curFam) : "n";
    let name;
    if (source === "model") name = $("#f-model_name").value;
    else name = pickModel(fam.key, $("#f-task").value, size);
    S.state.model_name = name;
    queueSave({ model_name: name }, true);
  }

  function renderModelSource() {
    const custom = S.state.model_source === "custom";
    $("#std-model-row").hidden = custom;
    $("#custom-model-block").hidden = !custom;
    $$('input[name="model_source"]').forEach((r) => { r.checked = r.value === S.state.model_source; });
    $$('input[name="train_mode"]').forEach((r) => { r.checked = r.value === S.state.train_mode; });
    const t = S.meta.tasks.find((x) => x.value === S.state.task);
    $("#custom-task-label").textContent = t ? t.label : S.state.task;
  }

  // ---------------------------------------------------------------- состояние
  function renderState() {
    const st = S.state;
    renderModelSource();
    renderModelSelectors();
    ["model_path", "data", "project_dir", "name"].forEach((k) => {
      const el = document.getElementById("f-" + k);
      if (document.activeElement !== el) el.value = st[k] || "";
    });
    $("#f-profile").value = st.profile;
    Object.values(S.paramsByKey).forEach((p) => writeParam(p, st[p.key]));
    applyTaskVisibility();
    renderErrors();
  }

  function renderErrors() {
    $$(".field[data-key]").forEach((f) => {
      const msg = S.errors[f.dataset.key];
      f.classList.toggle("err", !!msg);
      const e = $(".err-text", f);
      if (e) { e.textContent = msg || ""; e.hidden = !msg; }
    });
    const other = Object.entries(S.errors).filter(([k]) => !document.querySelector(`.field[data-key="${k}"]`));
    if (other.length) showMessage(other.map(([, v]) => v).join("\n"), "error");
  }

  function queueSave(values, immediate) {
    Object.assign(S.pending, values);
    clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(flushSave, immediate ? 0 : SAVE_DEBOUNCE_MS);
  }

  async function flushSave() {
    clearTimeout(S.saveTimer);
    const values = S.pending;
    S.pending = {};
    if (!Object.keys(values).length) return;
    try {
      const res = await api("POST", "/api/state", values);
      S.state = res.state;
      S.errors = res.errors || {};
      showMessage("");
      renderState();
    } catch (e) {
      showMessage("Не удалось сохранить настройки: " + e.message, "error");
    }
  }

  // ---------------------------------------------------------------- действия
  async function browse(kind, inputId, key) {
    const input = document.getElementById(inputId);
    try {
      const res = await api("POST", "/api/browse", { kind, initial: input.value });
      if (res.path) {
        input.value = res.path;
        queueSave({ [key]: res.path }, true);
        return res.path;
      }
    } catch (e) {
      showMessage("Диалог выбора не открылся: " + e.message, "error");
    }
    return "";
  }

  async function inspectCustom() {
    await flushSave();
    const info = $("#custom-model-info");
    info.textContent = "Читаю файл…";
    try {
      const res = await api("POST", "/api/model/inspect", { path: $("#f-model_path").value });
      S.state = res.state;
      const classes = Object.values(res.classes || {});
      const shown = classes.slice(0, 12).join(", ") + (classes.length > 12 ? ", …" : "");
      info.textContent = `Классов: ${classes.length} (${shown})` +
        (res.imgsz ? ` · обучалась с imgsz=${res.imgsz}` : "") +
        (res.resumable ? " · можно продолжить прерванное обучение" : "");
      renderState();
    } catch (e) {
      info.textContent = "";
      showMessage("Файл не прочитан: " + e.message, "error");
    }
  }

  async function applyProfile() {
    const profile = $("#f-profile").value;
    try {
      const res = await api("GET", `/api/profile?task=${encodeURIComponent(S.state.task)}&profile=${encodeURIComponent(profile)}`);
      queueSave(Object.assign({ profile }, res.values), true);
      showMessage(`Подставлен профиль «${profile}» для задачи «${S.state.task}».`, "ok");
    } catch (e) {
      showMessage("Профиль не подставлен: " + e.message, "error");
    }
  }

  async function loadUserPresets(selected) {
    const res = await api("GET", "/api/user-presets");
    const sel = $("#f-user-preset");
    sel.innerHTML = '<option value="">— свои настройки —</option>';
    res.names.forEach((n) => {
      const o = document.createElement("option");
      o.value = n; o.textContent = n;
      sel.appendChild(o);
    });
    if (selected) sel.value = selected;
  }

  async function saveUserPreset() {
    const name = $("#f-user-preset-name").value.trim();
    if (!name) { showMessage("Введите имя для сохранения настроек.", "error"); return; }
    await flushSave();
    try {
      await api("POST", "/api/user-presets/save", { name });
      await loadUserPresets(name);
      showMessage(`Настройки сохранены: ${name}`, "ok");
    } catch (e) {
      showMessage("Не сохранено: " + e.message, "error");
    }
  }

  async function loadUserPreset() {
    const name = $("#f-user-preset").value;
    if (!name) return;
    try {
      const res = await api("POST", "/api/user-presets/load", { name });
      S.state = res.state;
      S.errors = {};
      renderState();
      showMessage(`Загружены настройки: ${name}`, "ok");
    } catch (e) {
      showMessage("Не загружено: " + e.message, "error");
    }
  }

  async function startJob(kind) {
    await flushSave();
    showMessage("");
    try {
      await api("POST", "/api/job/" + kind);
      S.rows = [];
      switchTab("progress");
      await pollStatus();
    } catch (e) {
      showMessage(e.message, "error");
    }
  }

  async function jobAction(action) {
    try {
      await api("POST", "/api/job/" + action);
      await pollStatus();
    } catch (e) {
      showError(e.message);
    }
  }

  function showError(text) {
    const el = $("#job-error");
    el.textContent = text || "";
    el.hidden = !text;
  }

  // ---------------------------------------------------------------- прогресс
  const STATE_LABELS = {
    starting: "запуск…", running: "идёт", finished: "завершено", stopped: "остановлено",
    killed: "прервано", failed: "ошибка",
  };

  async function pollStatus() {
    clearTimeout(S.pollTimer);
    let active = false;
    try {
      const res = await api("GET", "/api/job/status");
      S.job = res.job;
      S.rows = res.rows || [];
      renderJob(res.log || "");
      active = !!(S.job && S.job.active);
    } catch (e) {
      showError("Нет связи с сервером: " + e.message);
    }
    S.pollTimer = setTimeout(pollStatus, active ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  }

  function renderJob(log) {
    const job = S.job;
    const badge = $("#job-state");
    $("#tab-job-dot").hidden = !(job && job.active);
    $("#btn-stop").disabled = !(job && job.active && job.kind === "train");
    $("#btn-kill").disabled = !(job && job.active);
    $("#btn-open-run").disabled = !job;
    $("#btn-train").disabled = !!(job && job.active);
    $("#btn-validate").disabled = !!(job && job.active);
    if (!job) {
      badge.textContent = "нет запусков";
      badge.className = "badge";
      return;
    }
    badge.textContent = (job.kind === "validate" ? "проверка: " : "обучение: ") + (STATE_LABELS[job.state] || job.state);
    badge.className = "badge " + job.state;
    $("#job-dir").textContent = job.run_dir;
    const last = S.rows[S.rows.length - 1];
    $("#job-epoch").textContent = last ? `эпоха ${last.epoch} из ${last.epochs}` : "";
    showError(job.error);

    const logEl = $("#job-log");
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
    logEl.textContent = log;
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;

    renderCharts(job.task);
    renderLastMetrics(last);
  }

  function renderLastMetrics(row) {
    const box = $("#last-metrics");
    box.innerHTML = "";
    if (!row) return;
    Object.keys(row).filter((k) => k.startsWith("metrics/")).forEach((k) => {
      const div = document.createElement("div");
      const name = document.createElement("span");
      name.className = "k";
      name.textContent = k.replace("metrics/", "") + ": ";
      div.appendChild(name);
      div.appendChild(document.createTextNode(Number(row[k]).toFixed(4)));
      box.appendChild(div);
    });
  }

  function renderCharts(task) {
    const rows = S.rows;
    const keys = rows.length ? Object.keys(rows[rows.length - 1]) : [];
    const main = S.meta.main_metric[task];
    const metricKeys = keys.filter((k) => k.startsWith("metrics/") && (k === main || /mAP50\(|accuracy_top5/.test(k)));
    if (main && !metricKeys.includes(main) && keys.includes(main)) metricKeys.unshift(main);
    const lossKeys = keys.filter((k) => k.startsWith("train/") || k.startsWith("val/"));
    drawChart($("#chart-metrics"), rows, metricKeys);
    drawChart($("#chart-loss"), rows, lossKeys);
  }

  function drawChart(box, rows, keys) {
    const W = Math.max(300, box.clientWidth || 600), H = 210, L = 48, R = 10, T = 10, B = 24;
    const ns = "http://www.w3.org/2000/svg";
    box.innerHTML = "";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const el = (name, attrs, text) => {
      const n = document.createElementNS(ns, name);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      if (text !== undefined) n.textContent = text;
      svg.appendChild(n);
      return n;
    };
    if (!rows.length || !keys.length) {
      el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" }, "данные появятся после первой эпохи");
      box.appendChild(svg);
      return;
    }
    const xs = rows.map((r) => r.epoch);
    const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);
    let yMin = Infinity, yMax = -Infinity;
    keys.forEach((k) => rows.forEach((r) => {
      if (typeof r[k] === "number") { yMin = Math.min(yMin, r[k]); yMax = Math.max(yMax, r[k]); }
    }));
    if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    if (yMax === yMin) { yMax = yMin + 1; }
    const sx = (x) => L + (x - xMin) / (xMax - xMin) * (W - L - R);
    const sy = (y) => T + (1 - (y - yMin) / (yMax - yMin)) * (H - T - B);

    el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "axis" });
    el("line", { x1: L, y1: T, x2: L, y2: H - B, class: "axis" });
    for (let i = 0; i <= 4; i++) {
      const v = yMin + (yMax - yMin) * i / 4;
      el("text", { x: L - 4, y: sy(v) + 4, "text-anchor": "end", class: "tick" }, v.toFixed(3));
    }
    el("text", { x: L, y: H - 6, class: "tick" }, String(xMin));
    el("text", { x: W - R, y: H - 6, "text-anchor": "end", class: "tick" }, "эпоха " + xMax);

    keys.forEach((k, i) => {
      const pts = rows.filter((r) => typeof r[k] === "number").map((r) => `${sx(r.epoch).toFixed(1)},${sy(r[k]).toFixed(1)}`);
      if (!pts.length) return;
      el("polyline", { points: pts.join(" "), fill: "none", stroke: SERIES_COLORS[i % SERIES_COLORS.length],
        "stroke-width": 2, "vector-effect": "non-scaling-stroke" });
    });
    box.appendChild(svg);

    const legend = document.createElement("div");
    legend.className = "legend";
    keys.forEach((k, i) => {
      const s = document.createElement("span");
      const sw = document.createElement("i");
      sw.style.background = SERIES_COLORS[i % SERIES_COLORS.length];
      s.appendChild(sw);
      s.appendChild(document.createTextNode(k));
      legend.appendChild(s);
    });
    box.appendChild(legend);
  }

  // ---------------------------------------------------------------- привязка событий
  function bindEvents() {
    $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

    $$('input[name="model_source"]').forEach((r) => r.addEventListener("change", () => {
      S.state.model_source = r.value;
      renderModelSource();
      queueSave({ model_source: r.value }, true);
    }));
    $$('input[name="train_mode"]').forEach((r) => r.addEventListener("change", () => queueSave({ train_mode: r.value }, true)));
    $("#f-family").addEventListener("change", () => onModelSelectorChange("family"));
    $("#f-task").addEventListener("change", () => onModelSelectorChange("task"));
    $("#f-model_name").addEventListener("change", () => onModelSelectorChange("model"));

    ["model_path", "data", "project_dir", "name"].forEach((k) => {
      document.getElementById("f-" + k).addEventListener("change", (e) => queueSave({ [k]: e.target.value }));
    });
    $("#f-model_path").addEventListener("change", () => setTimeout(inspectCustom, 0));

    $("#btn-browse-weights").addEventListener("click", async () => {
      if (await browse("weights", "f-model_path", "model_path")) inspectCustom();
    });
    $("#btn-inspect").addEventListener("click", inspectCustom);
    $("#btn-browse-data-yaml").addEventListener("click", () => browse("yaml", "f-data", "data"));
    $("#btn-browse-data-dir").addEventListener("click", () => browse("folder", "f-data", "data"));
    $("#btn-browse-project").addEventListener("click", () => browse("folder", "f-project_dir", "project_dir"));
    $("#btn-open-project").addEventListener("click", () => api("POST", "/api/open", { path: $("#f-project_dir").value })
      .catch((e) => showMessage(e.message, "error")));
    $("#btn-open-run").addEventListener("click", () => S.job && api("POST", "/api/open", { path: S.job.run_dir })
      .catch((e) => showError(e.message)));

    $("#f-profile").addEventListener("change", (e) => queueSave({ profile: e.target.value }));
    $("#btn-apply-profile").addEventListener("click", applyProfile);
    $("#btn-save-preset").addEventListener("click", saveUserPreset);
    $("#btn-load-preset").addEventListener("click", loadUserPreset);
    $("#btn-train").addEventListener("click", () => startJob("train"));
    $("#btn-validate").addEventListener("click", () => startJob("validate"));
    $("#btn-stop").addEventListener("click", () => jobAction("stop"));
    $("#btn-kill").addEventListener("click", () => {
      if (window.confirm("Прервать обучение немедленно? Текущая эпоха будет потеряна.")) jobAction("kill");
    });
  }

  // ---------------------------------------------------------------- старт
  async function init() {
    initTooltip();
    bindEvents();
    try {
      S.meta = await api("GET", "/api/meta");
      fillSelect($("#f-profile"), S.meta.profiles);
      buildParamGroups();
      const st = await api("GET", "/api/state");
      S.state = st.state;
      renderState();
      await loadUserPresets();
      const help = await api("GET", "/api/help");
      $("#help-content").innerHTML = Object.values(help).join("\n");
    } catch (e) {
      showMessage("Ошибка загрузки интерфейса: " + e.message, "error");
    }
    pollStatus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
