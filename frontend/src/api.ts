// API v1.0: приложение и API находятся на одном origin (/ui/ и /classify).
export type Status = 'ok' | 'review' | 'not_found';
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed';
export type Verdict = 'correct' | 'incorrect' | 'absent';

export type Alternative = {
  code: string | null;
  title: string | null;
  confidence: number;
  clauseRef: string | null;
  url: string | null;
};
<<<<<<< Updated upstream
=======
export type ClarificationOption = {
  okpd2Title?: string | null;
  okpd2Context?: string | null;
  okpd2Source?: string | null;
  optionId: string;
  code: string;
  title: string;
  confidence: number;
  clauseRef: string | null;
  url: string | null;
  codeRole: string | null;
  catalogReviewRequired: boolean | null;
};
export type OriginalDecision = {
  code: string | null;
  title: string | null;
  status: Exclude<Status, 'needs_clarification'>;
  confidence: number;
  confidenceRaw: number | null;
};
export type Clarification = {
  question: string;
  options: ClarificationOption[];
  catalogSha256: string;
  original: OriginalDecision;
};
export type Resolution = {
  optionId: string;
  resolvedAt: string;
  original: OriginalDecision;
};
export type AttentionReason = { code: string; message: string };
export type DecisionSource = 'model' | 'specialist' | 'direct_code';
>>>>>>> Stashed changes
export type Result = {
  name: string;
  code: string | null;
  title: string | null;
  confidence: number | null; // калиброванная уверенность, 0..1
  confidenceRaw: number | null; // технический скор до калибровки
  status: Status;
  clauseRef: string | null;
  url: string | null;
  alternatives: Alternative[];
  codeRole?: string | null;
  catalogReviewRequired?: boolean | null;
  userVerdict?: Verdict | null;
  correctedCode?: string | null;
  // Провенанс ответа, если его подтвердила проверяющая ступень (LLM по top-3):
  // "llm-top3" и короткая пометка. null/undefined — ответ дан классификатором.
  verifier?: string | null;
  verifierNote?: string | null;
};
export type Job = {
  job_id: string;
  status: JobStatus;
  processed: number;
  total: number;
  results: Result[];
  error: string | null;
};
export type UploadAccepted = {
  job_id: string;
  filename: string;
  format: string;
  total: number;
  total_found: number;
  duplicates: number;
  skipped: number;
  truncated: boolean;
  source_column: string | null;
  warnings: string[];
};

type ApiAlternative = {
  code: string | null; title: string | null; confidence: number;
  clause_ref: string | null; clause_url: string | null;
};
<<<<<<< Updated upstream
=======
type ApiOriginalDecision = {
  code: string | null; title: string | null;
  status: Exclude<Status, 'needs_clarification'>;
  confidence: number; confidence_raw: number | null;
};
type ApiClarificationOption = {
  okpd2_title?: string | null;
  okpd2_context?: string | null;
  okpd2_source?: string | null;
  option_id: string; code: string; title: string; confidence: number;
  clause_ref: string | null; clause_url: string | null;
  code_role: string | null; catalog_review_required: boolean | null;
};
type ApiClarification = {
  question: string; options: ApiClarificationOption[]; catalog_sha256: string;
  original: ApiOriginalDecision;
};
type ApiResolution = { option_id: string; resolved_at: string; original: ApiOriginalDecision };
>>>>>>> Stashed changes
type ApiClassify = {
  query: string; code: string | null; title: string | null;
  confidence: number; confidence_raw: number | null; status: Status;
  clause_ref: string | null; clause_url: string | null;
  verifier?: string | null; verifier_note?: string | null;
  code_role?: string | null;
  catalog_review_required?: boolean | null;
  alternatives?: ApiAlternative[];
};
type ApiBatch = {
  job_id: string; status: JobStatus; total: number; completed: number;
  results: ApiClassify[]; error: string | null;
};
const REQUEST_TIMEOUT_MS = 30_000;

function describeError(status: number, body: unknown): string {
  const detail = body && typeof body === 'object' && 'detail' in body
    ? (body as { detail: unknown }).detail : undefined;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const errors = detail.map(item => item && typeof item === 'object' && 'msg' in item
      ? String((item as { msg: unknown }).msg) : '').filter(Boolean);
    if (errors.length) return `Проверьте данные: ${errors.join('; ')}`;
  }
  if (status === 413) return 'Файл слишком большой для сервера.';
  if (status === 415) return 'Формат файла не поддерживается.';
  if (status === 429) return 'Слишком много запросов. Подождите и попробуйте снова.';
  if (status === 503) return 'Классификатор временно недоступен. Попробуйте позже.';
  if (status === 404) return 'Задание не найдено или срок хранения истёк.';
  return `Ошибка сервера (HTTP ${status}).`;
}

async function request<T>(path: string, init: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      method: init.method ?? 'GET',
      headers: init.form ? { Accept: 'application/json' }
        : init.body === undefined ? { Accept: 'application/json' }
          : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: init.form ?? (init.body === undefined ? undefined : JSON.stringify(init.body)),
      signal: controller.signal,
    });
    const text = await response.text();
    let body: unknown = null;
    if (text) { try { body = JSON.parse(text); } catch { /* HTML and malformed JSON */ } }
    if (!response.ok) throw new Error(describeError(response.status, body));
    if (!body || typeof body !== 'object') throw new Error('Сервер вернул неожиданный ответ.');
    return body as T;
  } catch (error) {
    if (controller.signal.aborted) throw new Error('Сервер не ответил за 30 секунд. Попробуйте снова.');
    if (error instanceof TypeError) throw new Error('Нет связи с API. Проверьте, что сервер запущен.');
    throw error;
  } finally { clearTimeout(timer); }
}

function toResult(item: ApiClassify): Result {
  // Защита от противоречивого ответа backend: статус ok без кода не должен
  // выглядеть как уверенная рекомендация.
  const invalidCode = !item.code && item.status !== 'not_found';
  const notFound = item.status === 'not_found';
  return {
    name: item.query, code: notFound ? null : item.code,
    title: notFound ? null : item.title,
    confidence: invalidCode ? null : item.confidence,
    confidenceRaw: notFound ? null : item.confidence_raw ?? null,
    status: invalidCode ? 'review' : item.status,
    clauseRef: notFound || invalidCode ? null : item.clause_ref,
    url: notFound || invalidCode ? null : item.clause_url,
    codeRole: item.code_role ?? null,
    catalogReviewRequired: item.catalog_review_required ?? null,
    verifier: item.verifier ?? null,
    verifierNote: item.verifier_note ?? null,
<<<<<<< Updated upstream
=======
    itemId: item.item_id ?? null,
    catalogSha256: item.catalog_sha256 ?? null,
    decisionSource,
    attentionReasons: item.attention_reasons ?? [],
    clarification: item.clarification ? {
      question: item.clarification.question,
      catalogSha256: item.clarification.catalog_sha256,
      original: toOriginal(item.clarification.original),
      options: item.clarification.options.map(option => ({
        optionId: option.option_id, code: option.code, title: option.title,
        confidence: option.confidence, clauseRef: option.clause_ref,
        url: option.clause_url, codeRole: option.code_role,
        catalogReviewRequired: option.catalog_review_required,
        okpd2Context: option.okpd2_context ?? null,
        okpd2Title: option.okpd2_title ?? null, okpd2Source: option.okpd2_source ?? null,
      })),
    } : null,
    resolution: item.resolution ? {
      optionId: item.resolution.option_id,
      resolvedAt: item.resolution.resolved_at,
      original: toOriginal(item.resolution.original),
    } : null,
>>>>>>> Stashed changes
    alternatives: (item.alternatives ?? []).slice(0, 3).map(alt => ({
      code: alt.code, title: alt.title, confidence: alt.confidence,
      clauseRef: alt.clause_ref, url: alt.clause_url,
    })),
  };
}
function toJob(raw: ApiBatch): Job {
  return {
    job_id: raw.job_id, status: raw.status, processed: raw.completed,
    total: raw.total, results: (raw.results ?? []).map(toResult), error: raw.error,
  };
}

export const api = {
  async classify(name: string): Promise<Result> {
    return toResult(await request<ApiClassify>('/classify', { method: 'POST', body: { name } }));
  },
  async uploadFile(file: File): Promise<UploadAccepted> {
    const form = new FormData();
    form.append('file', file);
    return request<UploadAccepted>('/classify/file', { method: 'POST', form });
  },
  async getBatch(jobId: string): Promise<Job> {
    return toJob(await request<ApiBatch>(`/classify/batch/${encodeURIComponent(jobId)}`));
  },
  async feedback(result: Result, verdict: Verdict, correctedCode?: string): Promise<void> {
    await request('/feedback', { method: 'POST', body: {
      query: result.name, predicted_code: result.code, verdict,
      corrected_code: verdict === 'incorrect' ? correctedCode?.trim() : null,
    } });
  },
};

/** SSE updates with polling fallback (e.g. offline intermediary blocking streams). */
export async function followBatch(
  id: string, onProgress: (job: Job) => void, signal: AbortSignal,
): Promise<Job> {
  signal.throwIfAborted();
  const deadline = Date.now() + 180_000;
  const eventUrl = `/classify/batch/${encodeURIComponent(id)}/events`;
  if ('EventSource' in window) {
    const done = await new Promise<boolean>((resolve, reject) => {
      const source = new EventSource(eventUrl);
      const cleanup = () => {
        window.clearTimeout(timeout);
        source.close();
        signal.removeEventListener('abort', abort);
      };
      const close = (value: boolean) => { cleanup(); resolve(value); };
      const abort = () => { cleanup(); reject(new DOMException('Запрос отменён', 'AbortError')); };
      const timeout = window.setTimeout(() => close(false), 140_000);
      if (signal.aborted) { abort(); return; }
      signal.addEventListener('abort', abort, { once: true });
      source.addEventListener('progress', event => {
        try {
          const v = JSON.parse((event as MessageEvent).data) as { status: JobStatus; total: number; completed: number };
          onProgress({ job_id: id, status: v.status, processed: v.completed, total: v.total, results: [], error: null });
        } catch { /* Ignore malformed SSE event; final poll is source of truth. */ }
      });
      source.addEventListener('done', () => close(true));
      source.addEventListener('error', () => close(false));
      source.onerror = () => close(false);
    });
    if (done && !signal.aborted) {
      const job = await api.getBatch(id);
      signal.throwIfAborted();
      onProgress(job);
      return job;
    }
  }
  // All browsers retain functional progress without SSE.
  while (!signal.aborted) {
    if (Date.now() > deadline) throw new Error('Ожидание превысило 3 минуты. Задание можно восстановить после обновления страницы.');
    const job = await api.getBatch(id);
    if (signal.aborted) break;
    onProgress(job);
    if (job.status === 'completed' || job.status === 'failed') return job;
    await new Promise<void>(resolve => {
      const finish = () => {
        window.clearTimeout(timer);
        signal.removeEventListener('abort', finish);
        resolve();
      };
      const timer = window.setTimeout(finish, 700);
      signal.addEventListener('abort', finish, { once: true });
      if (signal.aborted) finish();
    });
  }
  throw new DOMException('Запрос отменён', 'AbortError');
}
