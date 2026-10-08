/* Dora B2BCase demo client.
 *
 * Uses only the public API contract v1.0 (docs/api-contract.md):
 *   POST /classify        {name}                -> ClassifyResponse
 *   POST /classify/batch  {items[]}             -> {job_id, status_url, ...}
 *   GET  /classify/batch/{id}                   -> {status, progress_percent, results[]}
 * All URLs are relative so the UI works from the same origin as the API.
 */
"use strict";

const $ = (id) => document.getElementById(id);

function apiBase() {
  const custom = $("api-base").value.trim().replace(/\/+$/, "");
  return custom || window.location.origin;
}

function setStatus(text) {
  $("status-line").textContent = text || "";
}

async function api(path, options) {
  const response = await fetch(apiBase() + path, {
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
    setStatus(`Готово за ${result.latency_ms} мс.`);
  } catch (error) {
    setStatus(`Ошибка: ${error.message}`);
  }
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
  $("progress-wrap").classList.remove("hidden");
  $("progress-bar").style.width = "0%";
  try {
    const accepted = await api("/classify/batch", {
      method: "POST",
      body: JSON.stringify({ items }),
    });
    let statusUrl = accepted.status_url;
    if (!/^https?:/i.test(statusUrl)) {
      statusUrl = apiBase() + statusUrl;
    }
    // Poll the job until completion (the API contract exposes progress_percent).
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 350));
      const state = await fetch(statusUrl).then((r) => r.json());
      const percent = state.progress_percent || 0;
      $("progress-bar").style.width = `${percent}%`;
      setStatus(
        `Обработано ${state.completed} из ${state.total} (${Math.round(percent)}%)…`
      );
      if (state.status === "completed") {
        renderResults(state.results || []);
        setStatus(`Готово: ${state.total} позиций.`);
        break;
      }
      if (state.status === "failed") {
        setStatus(`Задача завершилась с ошибкой: ${state.error || "неизвестно"}`);
        break;
      }
    }
  } catch (error) {
    setStatus(`Ошибка: ${error.message}`);
  } finally {
    setTimeout(() => $("progress-wrap").classList.add("hidden"), 700);
  }
}

$("single-btn").addEventListener("click", classifySingle);
$("batch-btn").addEventListener("click", classifyBatch);
$("single-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter") classifySingle();
});
