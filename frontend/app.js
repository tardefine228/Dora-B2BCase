/* Dora B2BCase demo client.
 *
 * Uses only the public API contract v1.0 (docs/api-contract.md):
 *   POST /classify                        {name}    -> ClassifyResponse
 *   POST /classify/batch                  {items[]} -> {job_id, status_url, ...}
 *   POST /classify/file                   multipart -> {job_id, status_url, items[], ...}
 *   GET  /classify/batch/{id}                       -> {status, progress_percent, results[]}
 *   GET  /classify/batch/{id}/events                -> text/event-stream (progress/done)
 *   GET  /classify/batch/{id}.csv                   -> CSV (; + UTF-8 BOM)
 *   GET  /source/{record_id}                        -> fragment of the decree text
 * All URLs are relative so the UI works from the same origin as the API.
 */
"use strict";

const $ = (id) => document.getElementById(id);

/* CSV columns of a batch export; keep in sync with CSV_COLUMNS in api/app/main.py. */
const CSV_COLUMNS = [
  "query",
  "code",
  "title",
  "status",
  "confidence",
  "clause_ref",
  "clause_url",
  "alt1_code",
  "alt1_title",
  "alt2_code",
  "alt2_title",
  "alt3_code",
  "alt3_title",
];
const MAX_ALTERNATIVES = 3;
/* If SSE does not deliver a single event within this time, fall back to polling. */
const SSE_FIRST_EVENT_TIMEOUT_MS = 5000;
const POLL_INTERVAL_MS = 350;

let lastBatch = null; // {jobId, results} of the last finished batch

function apiBase() {
  const custom = $("api-base").value.trim().replace(/\/+$/, "");
  return custom || window.location.origin;
}

function apiUrl(path) {
  return /^https?:/i.test(path) ? path : apiBase() + path;
}

function setStatus(text) {
  $("status-line").textContent = text || "";
}

async function api(path, options) {
  const response = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch (_err) {
    /* non-JSON error body */
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function badge(status) {
  const labels = { ok: "ok", review: "review", not_found: "not_found" };
  return `<span class="badge ${status}">${labels[status] || status}</span>`;
}

function confidenceBar(value) {
  const percent = Math.round((value || 0) * 100);
  return `
    <div class="conf" title="${percent}%">
      <div class="conf-fill" style="width:${percent}%"></div>
      <span class="conf-label">${percent}%</span>
    </div>`;
}

function sourceLink(clauseUrl) {
  if (!clauseUrl) return "";
  return `<a class="source-link" href="${escapeHtml(apiUrl(clauseUrl))}" target="_blank"
    rel="noopener">источник</a>`;
}

function alternativesHtml(alternatives) {
  if (!alternatives || !alternatives.length) return "";
  const items = alternatives
    .map(
      (alt) => `
      <li>
        <code>${escapeHtml(alt.code || "—")}</code>
        ${escapeHtml(alt.title || "")}
        <span class="muted">${Math.round((alt.confidence || 0) * 100)}%</span>
        ${alt.clause_ref ? `<span class="muted">· ${escapeHtml(alt.clause_ref)}</span>` : ""}
        ${alt.clause_url ? `· ${sourceLink(alt.clause_url)}` : ""}
      </li>`
    )
    .join("");
  return `<details><summary>Альтернативы (${alternatives.length})</summary><ul class="alts">${items}</ul></details>`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = String(text ?? "");
  return div.innerHTML;
}

function resultHtml(result) {
  return `
    <article class="result">
      <div class="result-head">
        <strong>${escapeHtml(result.query)}</strong>
        ${badge(result.status)}
      </div>
      <div class="result-body">
        <div class="code">${escapeHtml(result.code || "— не найдено —")}</div>
        <div class="title">${escapeHtml(result.title || "")}</div>
        ${confidenceBar(result.confidence)}
        ${result.clause_ref ? `<div class="clause">${escapeHtml(result.clause_ref)}</div>` : ""}
        ${result.clause_url ? `<div class="clause">${sourceLink(result.clause_url)}</div>` : ""}
        ${alternativesHtml(result.alternatives)}
      </div>
    </article>`;
}

function renderResults(results) {
  const container = $("results");
  if (!results.length) {
    container.innerHTML = "";
    $("empty-hint").classList.remove("hidden");
    return;
  }
  $("empty-hint").classList.add("hidden");
  container.innerHTML = results.map(resultHtml).join("");
}

/* --- CSV export --------------------------------------------------------- */

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[";\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/* Excel on a Russian locale wants ";" separators, CRLF rows and a UTF-8 BOM. */
function buildCsv(results) {
  const lines = [CSV_COLUMNS.join(";")];
  results.forEach((result) => {
    const alternatives = (result.alternatives || []).slice(0, MAX_ALTERNATIVES);
    const row = [
      result.query,
      result.code || "",
      result.title || "",
      result.status,
      Number(result.confidence || 0).toFixed(4),
      result.clause_ref || "",
      result.clause_url || "",
    ];
    alternatives.forEach((alt) => row.push(alt.code || "", alt.title || ""));
    while (row.length < CSV_COLUMNS.length) row.push("");
    lines.push(row.map(csvCell).join(";"));
  });
  return "\ufeff" + lines.join("\r\n") + "\r\n";
}

function setCsvControls(jobId, results) {
  const button = $("csv-btn");
  const serverLink = $("csv-server-link");
  if (!results || !results.length) {
    button.classList.add("hidden");
    serverLink.classList.add("hidden");
    return;
  }
  button.classList.remove("hidden");
  button.title = `Экспорт ${results.length} строк в CSV (разделитель «;», UTF-8 BOM)`;
  if (jobId) {
    serverLink.href = apiUrl(`/classify/batch/${jobId}.csv`);
    serverLink.classList.remove("hidden");
  }
}

function downloadBatchCsv() {
  if (!lastBatch || !lastBatch.results.length) return;
  const blob = new Blob([buildCsv(lastBatch.results)], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `batch-${lastBatch.jobId}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus(`CSV готов: ${lastBatch.results.length} строк.`);
}

/* --- batch progress: SSE with polling fallback -------------------------- */

function progressText(state) {
  const percent = state.progress_percent ?? 0;
  return `Обработано ${state.completed} из ${state.total} (${Math.round(percent)}%)…`;
}

async function pollBatch(statusUrl, handlers) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    let state;
    try {
      const response = await fetch(apiUrl(statusUrl));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state = await response.json();
    } catch (error) {
      handlers.onError(error);
      return;
    }
    handlers.onProgress(state);
    if (state.status === "completed" || state.status === "failed") {
      handlers.onDone(state);
      return;
    }
  }
}

function watchBatch(statusUrl, handlers) {
  let finished = false;
  const done = (state) => {
    if (finished) return;
    finished = true;
    handlers.onDone(state);
  };

  if (typeof EventSource === "undefined") {
    pollBatch(statusUrl, handlers);
    return;
  }

  const source = new EventSource(apiUrl(`${statusUrl}/events`));
  let sawEvent = false;
  // A proxy may buffer the stream: if nothing arrives quickly, poll instead.
  const fallbackTimer = setTimeout(() => {
    if (finished || sawEvent) return;
    source.close();
    pollBatch(statusUrl, handlers);
  }, SSE_FIRST_EVENT_TIMEOUT_MS);

  source.addEventListener("progress", (event) => {
    sawEvent = true;
    try {
      handlers.onProgress(JSON.parse(event.data));
    } catch (error) {
      handlers.onError(error);
    }
  });
  source.addEventListener("done", (event) => {
    sawEvent = true;
    clearTimeout(fallbackTimer);
    source.close();
    let state = null;
    try {
      state = JSON.parse(event.data);
    } catch (_err) {
      /* handled below */
    }
    if (state) {
      done(state);
    } else {
      pollBatch(statusUrl, handlers);
    }
  });
  source.addEventListener("error", () => {
    // EventSource fires "error" when the server closes the stream as well.
    clearTimeout(fallbackTimer);
    source.close();
    if (!finished) pollBatch(statusUrl, handlers);
  });
}

function collectBatchItems() {
  return $("batch-input")
    .value.split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

async function classifySingle() {
  const name = $("single-input").value.trim();
  if (!name) {
    setStatus("Введите название товара.");
    return;
  }
  setStatus("Классифицирую…");
  try {
    const result = await api("/classify", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    renderResults([result]);
    lastBatch = null;
    setCsvControls(null, null);
    setStatus(`Готово за ${result.latency_ms} мс.`);
  } catch (error) {
    setStatus(`Ошибка: ${error.message}`);
  }
}

/* Start a batch (via `start()` returning BatchJobAccepted) and render progress. */
async function runBatch(start) {
  $("progress-wrap").classList.remove("hidden");
  $("progress-bar").style.width = "0%";
  setCsvControls(null, null);
  try {
    const accepted = await start();
    const statusUrl = accepted.status_url;

    await new Promise((resolve) => {
      watchBatch(statusUrl, {
        onProgress: (state) => {
          const percent = state.progress_percent || 0;
          $("progress-bar").style.width = `${percent}%`;
          setStatus(progressText(state));
        },
        onDone: (state) => {
          $("progress-bar").style.width = "100%";
          const results = state.results || [];
          renderResults(results);
          if (state.status === "completed") {
            lastBatch = { jobId: state.job_id || accepted.job_id, results };
            setCsvControls(lastBatch.jobId, results);
            setStatus(`Готово: ${state.total} позиций. ${statusSummary(results)}`);
          } else {
            lastBatch = null;
            setCsvControls(null, null);
            setStatus(`Задача завершилась с ошибкой: ${state.error || "неизвестно"}`);
          }
          resolve();
        },
        onError: (error) => {
          setStatus(`Ошибка: ${error.message}`);
          resolve();
        },
      });
    });
  } catch (error) {
    setStatus(`Ошибка: ${error.message}`);
  } finally {
    setTimeout(() => $("progress-wrap").classList.add("hidden"), 700);
  }
}

function statusSummary(results) {
  const counts = { ok: 0, review: 0, not_found: 0 };
  results.forEach((r) => {
    counts[r.status] = (counts[r.status] || 0) + 1;
  });
  return `ok: ${counts.ok}, review: ${counts.review}, not_found: ${counts.not_found}.`;
}

async function classifyBatch() {
  const items = collectBatchItems();
  if (!items.length) {
    setStatus("Добавьте хотя бы один товар в список.");
    return;
  }
  if (items.length > 100) {
    setStatus("Слишком много позиций: максимум 100 за запрос.");
    return;
  }
  setStatus(`Принято позиций: ${items.length}. Запускаю задачу…`);
  $("file-summary").classList.add("hidden");
  await runBatch(() =>
    api("/classify/batch", { method: "POST", body: JSON.stringify({ items }) })
  );
}

/* --- file upload -------------------------------------------------------- */

const MAX_UPLOAD_BYTES = 10_000_000;
const FILE_EXTENSIONS = [".txt", ".md", ".csv", ".tsv", ".json", ".xlsx", ".xlsm", ".docx", ".pdf"];

function renderFileSummary(accepted) {
  const box = $("file-summary");
  const parts = [
    `<strong>${escapeHtml(accepted.filename)}</strong> (${escapeHtml(accepted.format)}):
     найдено позиций — ${accepted.total_found}, отправлено на классификацию — ${accepted.total}`,
  ];
  if (accepted.source_column) parts.push(`источник: «${escapeHtml(accepted.source_column)}»`);
  if (accepted.duplicates) parts.push(`дубликатов пропущено — ${accepted.duplicates}`);
  if (accepted.skipped) parts.push(`пустых/служебных строк — ${accepted.skipped}`);
  const warnings = (accepted.warnings || [])
    .map((w) => `<li class="warn">${escapeHtml(w)}</li>`)
    .join("");
  box.innerHTML = parts.join(" · ") + (warnings ? `<ul>${warnings}</ul>` : "");
  box.classList.remove("hidden");
}

async function uploadFile(file) {
  if (!file) return;
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (!FILE_EXTENSIONS.includes(ext)) {
    setStatus(`Формат «${ext || "без расширения"}» не поддерживается.`);
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    setStatus("Файл больше 10 МБ.");
    return;
  }
  $("file-name").textContent = file.name;
  $("file-summary").classList.add("hidden");
  $("dropzone").classList.add("busy");
  setStatus(`Разбираю файл «${file.name}»…`);
  try {
    await runBatch(async () => {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(apiUrl("/classify/file"), { method: "POST", body: form });
      let body = null;
      try {
        body = await response.json();
      } catch (_err) {
        /* non-JSON error body */
      }
      if (!response.ok) {
        const detail = body && body.detail ? body.detail : response.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      renderFileSummary(body);
      setStatus(`Из файла извлечено ${body.total} позиций. Классифицирую…`);
      return body;
    });
  } finally {
    $("dropzone").classList.remove("busy");
    $("file-input").value = "";
  }
}

(function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => uploadFile(input.files[0]));
  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    })
  );
  zone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
})();

$("single-btn").addEventListener("click", classifySingle);
$("batch-btn").addEventListener("click", classifyBatch);
$("csv-btn").addEventListener("click", downloadBatchCsv);
$("single-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") classifySingle();
});
