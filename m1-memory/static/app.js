const state = {
  conv: null,
  empty: null,
  threads: [],
  meta: null,
  running: false,
  poll: null,
  drawerOpen: false,
};

const $ = (sel) => document.querySelector(sel);
const FLAGS = new URLSearchParams(location.search);
const EXPAND = FLAGS.has("expand");

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

const fmt = (value, digits = 3) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
const fmtInt = (value) =>
  typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US") : "—";

function toast(message, ms = 2600) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), ms);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// ------------------------------------------------------------------ sidebar

async function loadThreads() {
  const data = await api("/api/threads");
  state.threads = data.threads || [];
  const list = $("#thread-list");
  list.replaceChildren();
  for (const thread of state.threads) {
    const button = el("button", {
      class: `thread-item${thread.id === state.conv ? " active" : ""}`,
      onclick: () => openThread(thread.id),
    }, [
      el("b", { text: thread.preview || "（空会话）" }),
      el("span", { text: `${thread.messages} 条 · ${thread.id}` }),
    ]);
    list.append(button);
  }
  if (!state.threads.length) {
    list.append(el("p", { class: "note", text: "还没有会话，发一句话就开一个。" }));
  }
}

function paintStats(memory, encoder) {
  $("#stat-chunks").textContent = fmtInt(memory.chunks);
  $("#stat-docs").textContent = fmtInt(memory.docs);
  $("#stat-edges").textContent = fmtInt(memory.edges);
  $("#stat-encoder").textContent = encoder ? `${encoder.id.split("/").pop()}·${encoder.dim}` : "—";
}

// ------------------------------------------------------------------ thread

function openThread(id) {
  state.conv = id;
  if (state.empty) state.empty.style.display = "none";
  loadThreads();
  loadThread();
  loadMemory();
  $(".rail").classList.remove("open");
}

async function loadThread() {
  if (!state.conv) return;
  const data = await api(`/api/threads/${encodeURIComponent(state.conv)}`);
  const container = $("#thread");
  const previous = container.querySelector(".thread-inner");
  const scroller = container.scrollTop + container.clientHeight >= container.scrollHeight - 80;

  const inner = el("div", { class: "thread-inner" });
  for (const turn of data.turns) inner.append(renderTurn(turn));
  if (state.running) inner.append(renderLive(data.live || {}));

  if (!data.turns.length && !state.running) {
    container.replaceChildren(state.empty || inner);
    if (state.empty) state.empty.style.display = "block";
  } else if (previous) {
    previous.replaceWith(inner);
  } else {
    container.replaceChildren(inner);
  }

  $("#conv-title").textContent = data.title || state.conv;
  $("#conv-sub").textContent = `${state.conv} · ${data.turns.length} 轮`;
  if (scroller || state.running) container.scrollTop = container.scrollHeight;

  if (data.active && !state.running) {
    state.running = true;
    $("#send").disabled = true;
    pollRun(data.active);
  }
}

function renderTurn(turn) {
  const wrap = el("div", { class: "turn" });
  for (const message of turn.messages) {
    if (message.role === "user") {
      wrap.append(el("div", { class: "msg user" }, [el("div", { class: "bubble", text: message.content })]));
    } else {
      wrap.append(el("div", { class: "msg agent" }, [el("div", { class: "bubble", text: message.content })]));
    }
  }
  const chain = renderChain(turn.prepare || {}, turn.finalize || {}, turn.search || {});
  if (chain) wrap.append(chain);
  return wrap;
}

function renderLive(live) {
  const text = live.stage ? `${live.stage} · ${live.elapsed}s` : "正在跑这一轮";
  return el("div", { class: "thinking" }, [el("span", { class: "dot" }), el("span", { id: "live-stage", text })]);
}

// ------------------------------------------------------------------ chain

function stage(name, note, bodyNodes, open = false) {
  return el("details", { class: "stage", open: open || EXPAND ? "" : null }, [
    el("summary", {}, [
      el("span", { class: "stage-name", text: name }),
      el("span", { class: "stage-note", text: note || "" }),
    ]),
    el("div", { class: "stage-body" }, bodyNodes),
  ]);
}

function kv(rows) {
  const list = el("dl", { class: "kv" });
  for (const [key, value] of rows) {
    if (value === null || value === undefined || value === "") continue;
    list.append(el("dt", { text: key }), el("dd", { text: String(value) }));
  }
  return list;
}

function hits(rows, accent = "") {
  if (!rows || !rows.length) return el("p", { class: "note", text: "没有命中" });
  const max = Math.max(...rows.map((row) => row.score || 0), 1e-9);
  const box = el("div", {});
  for (const row of rows) {
    box.append(el("div", { class: "hit" }, [
      el("span", { class: "id", text: `#${row.chunk}` }),
      el("span", { class: "txt", text: row.preview || "" }),
      el("span", { class: "num", text: fmt(row.score, 3) }),
      el("div", { class: `bar ${accent}` }, [
        el("i", { style: `width:${Math.max(2, (row.score / max) * 100)}%` }),
      ]),
    ]));
  }
  return box;
}

function renderSearchStages(search) {
  const calls = (search && search.calls) || [];
  const nodes = [];
  calls.forEach((call, index) => {
    const results = call.results || [];
    const box = el("div", {});
    box.append(kv([
      ["关键词", call.query],
      ["引擎", call.engine || "tavily"],
      ["耗时", call.ms !== undefined ? `${call.ms} ms` : ""],
      ["结果", `${call.count || 0} 条`],
      ["错误", call.error],
    ]));
    for (const item of results) {
      box.append(el("div", { class: "webhit" }, [
        el("a", { class: "webhit-title", href: item.url, target: "_blank", rel: "noopener noreferrer", text: item.title || item.url }),
      ]));
    }
    if (!results.length) box.append(el("p", { class: "note", text: "这次没有可用结果" }));
    nodes.push(stage(`联网搜索 ${calls.length > 1 ? `#${index + 1}` : ""}`.trim(), `${call.query || ""} · ${call.count || 0} 条`, [box], true));
  });
  return nodes;
}

function renderChain(prepare, finalize, search = {}) {
  const hasData = prepare && Object.keys(prepare).length;
  const searchCalls = (search && search.calls) || [];
  if (!hasData) {
    if (!searchCalls.length) return null;
    return el("details", { class: "chain", open: EXPAND ? "" : null }, [
      el("summary", {}, [
        el("span", { class: "pulse" }),
        el("span", { class: "summary-main", text: "运行链" }),
        el("span", { class: "summary-meta" }, [el("span", { text: `${searchCalls.length} 次搜索` })]),
      ]),
      el("div", { class: "stages" }, renderSearchStages(search)),
    ]);
  }

  if (prepare.skipped && !prepare.retrieved) {
    return el("details", { class: "chain" }, [
      el("summary", {}, [
        el("span", { class: "pulse" }),
        el("span", { class: "summary-main", text: "未检索" }),
        el("span", { class: "summary-meta" }, [el("span", { text: prepare.skipped })]),
      ]),
      el("div", { class: "stages" }, renderSearchStages(search)),
    ]);
  }

  const diffusion = prepare.diffusion || {};
  const retrieved = prepare.retrieved || [];
  const gate = prepare.gate || {};
  const summary = el("summary", {}, [
    el("span", { class: "pulse" }),
    el("span", { class: "summary-main", text: "运行链" }),
    el("span", { class: "summary-meta" }, [
      el("span", { text: `${retrieved.length} 块` }),
      el("span", { text: `γ ${fmt(gate.gamma, 2)}` }),
      el("span", { text: `${diffusion.iterations || 0} 步` }),
      el("span", { text: searchCalls.length ? `搜索 ${searchCalls.length} 次` : "" }),
      el("span", { text: prepare.encode ? `${prepare.encode.ms}ms` : "" }),
    ]),
    el("span", { class: "summary-toggle", text: "展开" }),
  ]);

  const stages = el("div", { class: "stages" });
  const encode = prepare.encode || {};

  stages.append(stage("查询编码", `${encode.encoder || "?"} · ${encode.dim || "?"}d`, [
    kv([
      ["查询", prepare.query],
      ["编码器", `${encode.encoder || "—"} / ${encode.version || "—"}`],
      ["维度", encode.dim],
      ["耗时", encode.ms !== undefined ? `${encode.ms} ms` : ""],
      ["回落", encode.fallback ? "是（字符哈希）" : ""],
      ["错误", encode.error],
    ]),
  ], true));

  const lexical = prepare.lexical || {};
  const semantic = prepare.semantic || {};
  stages.append(stage("词法 BM25", `${lexical.candidates || 0} 条候选`, [
    el("div", { class: "chips" }, [
      ...(prepare.terms || []).slice(0, 14).map((term) => el("span", { class: "chip", text: term })),
      (prepare.terms || []).length > 14
        ? el("span", { class: "chip", text: `+${prepare.terms.length - 14}` })
        : null,
    ]),
    hits((lexical.hits || []).slice(0, 8), "rust"),
  ]));

  stages.append(stage("语义", `${semantic.candidates || 0} 条候选`, [
    hits((semantic.hits || []).slice(0, 8)),
  ]));

  const fused = prepare.fused || {};
  stages.append(stage("融合 d", `θ ${fmt(0.65, 2)} · ${prepare.route || ""}`, [
    kv([["d₁", fmt(fused.d1, 4)], ["d₂", fmt(fused.d2, 4)], ["路线", prepare.route]]),
    hits((fused.hits || []).slice(0, 6), "rust"),
  ]));

  stages.append(stage("门控 γ", `γ ${fmt(gate.gamma, 4)}`, [
    kv([
      ["有历史指针", gate.has_history ? "是" : "否"],
      ["ρ / τ", `${gate.rho} / ${gate.tau}`],
      ["γ", fmt(gate.gamma, 4)],
      ["上轮指针", (gate.previous_top || []).map((row) => `#${row.chunk} ${fmt(row.value, 3)}`).join("  ")],
    ]),
  ]));

  stages.append(stage("扩散激活", `${diffusion.iterations || 0} 步 · 残差 ${diffusion.residual || "—"}`, [
    kv([
      ["α", diffusion.alpha],
      ["迭代", diffusion.iterations],
      ["残差 L1", diffusion.residual],
      ["激活质量", fmt(diffusion.activation_mass, 4)],
      ["种子质量", fmt((prepare.seed || {}).mass, 4)],
      ["矩阵", prepare.matrix ? `${prepare.matrix.chunks} 块 / ${prepare.matrix.nonzeros} 非零 / ${prepare.matrix.edges} 邻接` : ""],
    ]),
    hits((diffusion.activation_top || []).map((row) => ({ chunk: row.neuron, score: row.value, preview: "神经元激活" })).slice(0, 6)),
  ]));

  const retrievedBox = el("div", {});
  for (const row of retrieved) {
    retrievedBox.append(el("div", { class: "retrieved" }, [
      el("div", { class: "head" }, [
        el("span", { text: `#${row.chunk}` }),
        el("span", { text: row.time }),
        el("span", { text: `total ${fmt(row.total)}` }),
        el("span", { text: `direct ${fmt(row.direct)}` }),
        el("span", { text: `neural ${fmt(row.neural)}` }),
      ]),
      el("div", { class: "text", text: row.preview || "" }),
    ]));
  }
  stages.append(stage("取回 K=5", `${retrieved.length} 块进入上下文`, [
    retrievedBox.children.length ? retrievedBox : el("p", { class: "note", text: "没有块进入上下文" }),
  ], true));

  for (const node of renderSearchStages(search)) stages.append(node);

  if (finalize && finalize.appended) {
    const memory = finalize.memory || {};
    stages.append(stage("写入记忆", `+${finalize.appended.length} 文档 · 共 ${memory.chunks} 块`, [
      kv([
        ["写入", finalize.appended.map((item) => `${item.doc_id}（${item.chunks} 块）`).join("；")],
        ["记忆库", `${memory.docs} 文档 / ${memory.chunks} 块 / ${memory.edges} 邻接`],
      ]),
    ]));
  }

  return el("details", { class: "chain", open: EXPAND ? "" : null }, [summary, stages]);
}

// ------------------------------------------------------------------ send

async function send() {
  const input = $("#input");
  const content = input.value.trim();
  if (!content || state.running) return;
  if (!state.conv) state.conv = `c-${Math.random().toString(36).slice(2, 10)}`;

  input.value = "";
  input.style.height = "auto";
  state.running = true;
  $("#send").disabled = true;
  $("#status").textContent = "排队中…";

  const container = $("#thread");
  let inner = container.querySelector(".thread-inner");
  if (!inner) {
    inner = el("div", { class: "thread-inner" });
    container.replaceChildren(inner);
  }
  if (state.empty) state.empty.style.display = "none";
  inner.append(el("div", { class: "turn" }, [
    el("div", { class: "msg user" }, [el("div", { class: "bubble", text: content })]),
    renderLive({}),
  ]));
  container.scrollTop = container.scrollHeight;

  try {
    const data = await api(`/api/threads/${encodeURIComponent(state.conv)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    pollRun(data.run);
  } catch (error) {
    finishRun();
    toast(error.message);
  }
}

function pollRun(runId) {
  clearInterval(state.poll);
  const tick = async () => {
    try {
      const run = await api(`/api/runs/${runId}`);
      const live = run.live || {};
      const label = $("#live-stage");
      if (label) label.textContent = live.stage ? `${live.stage} · ${live.elapsed}s` : "正在跑这一轮";
      $("#status").textContent = live.stage || "运行中";

      if (run.status === "done") {
        clearInterval(state.poll);
        finishRun();
        await loadThread();
        await loadThreads();
        await loadMemory();
        $("#status").textContent = `完成 · ${run.ms} ms · revision ${run.revision}`;
      } else if (run.status === "error") {
        clearInterval(state.poll);
        finishRun();
        await loadThread();
        const inner = $("#thread").querySelector(".thread-inner");
        if (inner) {
          inner.append(el("div", { class: "turn" }, [
            el("div", { class: "msg agent failed" }, [el("div", { class: "bubble", text: `这一轮失败了：${run.error}` })]),
          ]));
        }
        $("#status").textContent = "失败";
      }
    } catch (error) {
      clearInterval(state.poll);
      finishRun();
      toast(error.message);
    }
  };
  tick();
  state.poll = setInterval(tick, 600);
}

function finishRun() {
  state.running = false;
  $("#send").disabled = false;
  const label = $("#live-stage");
  if (label && label.parentElement) label.parentElement.remove();
}

// ------------------------------------------------------------------ memory

async function loadMemory() {
  const query = state.conv ? `?conversation=${encodeURIComponent(state.conv)}` : "";
  const data = await api(`/api/memory${query}`);
  paintStats(data.memory, data.encoder);
  $("#chunk-count").textContent = data.memory.chunks;
  $("#drawer-sub").textContent = `${data.memory.docs} 文档 · ${data.memory.chunks} 块 · ${data.memory.edges} 邻接 · ${data.encoder ? data.encoder.id : "?"}`;

  const list = $("#chunk-list");
  list.replaceChildren();
  const rows = (data.chunks || []).slice().reverse();
  if (!rows.length) {
    list.append(el("p", { class: "note", text: "记忆库还是空的。可以点上面的「预置示例记忆」，或者直接开始聊。" }));
    return;
  }
  for (const chunk of rows) {
    const max = Math.max(...rows.map((row) => row.activation || 0), 1e-9);
    list.append(el("div", { class: "chunk" }, [
      el("details", {}, [
        el("summary", {}, [
          el("div", { class: "meta" }, [
            el("span", { text: `#${chunk.i}` }),
            el("span", { text: chunk.time }),
            el("span", { text: chunk.activation ? `p ${fmt(chunk.activation, 3)}` : "" }),
          ]),
          el("div", { class: "text", text: chunk.text.replace(/\s+/g, " ").slice(0, 90) }),
          el("div", { class: "act" }, [el("i", { style: `width:${(chunk.activation / max) * 100}%` })]),
        ]),
        el("p", { class: "note", text: chunk.text }),
      ]),
    ]));
  }
}

// ------------------------------------------------------------------ events

$("#new-thread").addEventListener("click", () => {
  state.conv = `c-${Math.random().toString(36).slice(2, 10)}`;
  $("#thread").replaceChildren(state.empty);
  state.empty.style.display = "block";
  $("#conv-title").textContent = "新对话";
  $("#conv-sub").textContent = "";
  $("#input").focus();
  loadThreads();
  loadMemory();
  $(".rail").classList.remove("open");
});

$("#send").addEventListener("click", send);
$("#input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
$("#input").addEventListener("input", (event) => {
  const node = event.target;
  node.style.height = "auto";
  node.style.height = `${Math.min(node.scrollHeight, 180)}px`;
});

for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    $("#input").value = button.dataset.prompt;
    $("#input").focus();
    $("#input").dispatchEvent(new Event("input"));
  });
}

function toggleDrawer(open) {
  state.drawerOpen = open;
  $("#drawer").classList.toggle("open", open);
  $("#drawer").setAttribute("aria-hidden", open ? "false" : "true");
  if (open) loadMemory();
}

$("#open-memory").addEventListener("click", () => toggleDrawer(true));
$("#open-memory-2").addEventListener("click", () => toggleDrawer(true));
$("#close-drawer").addEventListener("click", () => toggleDrawer(false));
$("#drawer").addEventListener("click", (event) => {
  if (event.target === $("#drawer")) toggleDrawer(false);
});

$("#toggle-rail").addEventListener("click", () => $(".rail").classList.toggle("open"));

$("#seed").addEventListener("click", async () => {
  $("#seed").disabled = true;
  try {
    const data = await api("/api/memory/seed", { method: "POST", body: "{}" });
    toast(`写入 ${data.appended.length} 条示例记忆`);
    await loadMemory();
  } catch (error) {
    toast(error.message);
  } finally {
    $("#seed").disabled = false;
  }
});

$("#add-doc").addEventListener("click", async () => {
  const text = $("#doc-input").value.trim();
  if (!text) return;
  $("#add-doc").disabled = true;
  try {
    await api("/api/memory/docs", {
      method: "POST",
      body: JSON.stringify({ text, time: $("#doc-time").value.trim() || undefined }),
    });
    $("#doc-input").value = "";
    toast("已写入记忆");
    await loadMemory();
  } catch (error) {
    toast(error.message);
  } finally {
    $("#add-doc").disabled = false;
  }
});

$("#reset-pointer").addEventListener("click", async () => {
  if (!state.conv) return toast("先选一个会话");
  try {
    await api("/api/memory/reset-pointer", {
      method: "POST",
      body: JSON.stringify({ conversation: state.conv }),
    });
    toast("指针已重置");
    await loadMemory();
  } catch (error) {
    toast(error.message);
  }
});

(async function boot() {
  state.empty = document.getElementById("empty");
  try {
    state.meta = await api("/api/meta");
    await loadThreads();
    await loadMemory();
    if (state.threads.length) openThread(state.threads[0].id);
    if (FLAGS.has("drawer")) toggleDrawer(true);
  } catch (error) {
    toast(`初始化失败：${error.message}`);
  }
})();
