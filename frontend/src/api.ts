// Клиент API v1.0 (docs/api-contract.md).
// Адреса относительные: в Docker UI и API отдаются с одного origin, в `npm run dev`
// запросы проксирует vite (vite.config.ts, переменная API_PROXY_TARGET).

export type Status = 'ok' | 'review' | 'not_found';
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed';

export type Result = {
  confidenceKind?: 'calibrated' | 'uncalibrated' | 'exact';
  alternatives?: { code: string | null; title: string | null; clause_ref: string | null; clause_url: string | null }[];
  name: string;
  code: string | null;
  title: string | null;
  /** 0..1; null при not_found. */
  confidence: number | null;
  status: Status;
  /** Где в перечне: «Приложение 2, позиция 339». */
  clauseRef: string | null;
  /** Относительная ссылка `/source/{record_id}` (или абсолютная, если API вернёт её). */
  url: string | null;
};

export type Job = {
  job_id: string;
  status: JobStatus;
  /** Сколько позиций обработано (поле `completed` в API). */
  processed: number;
  total: number;
  results: Result[];
  error: string | null;
};

export interface ClassifierApi {
  classify(name: string): Promise<Result>;
  startBatch(names: string[]): Promise<{ job_id: string }>;
  getBatch(id: string): Promise<Job>;
}

// Ответы API в snake_case, как в docs/api-contract.md.
type ApiClassify = {
  confidence_kind?: 'calibrated' | 'uncalibrated' | 'exact';
  alternatives?: Result['alternatives'];
  query: string;
  code: string | null;
  title: string | null;
  confidence: number;
  confidence_raw: number | null;
  status: Status;
  clause_ref: string | null;
  clause_url: string | null;
};
type ApiBatch = {
  job_id: string;
  status: JobStatus;
  total: number;
  completed: number;
  progress_percent: number;
  results: ApiClassify[];
  error: string | null;
};
type ApiAccepted = { job_id: string; status: 'queued'; total: number; status_url: string };

const REQUEST_TIMEOUT_MS = 30_000;

function describeError(status: number, body: unknown): string {
  const detail = body && typeof body === 'object' && 'detail' in body ? (body as { detail: unknown }).detail : undefined;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    const messages = detail
      .map(item => (item && typeof item === 'object' && 'msg' in item ? String((item as { msg: unknown }).msg) : ''))
      .filter(Boolean);
    if (messages.length) return `Проверьте данные: ${messages.join('; ')}`;
  }
  if (status === 429) return 'Сервер сейчас занят. Подождите несколько секунд и попробуйте снова.';
  if (status === 503) return 'Классификатор временно недоступен. Попробуйте позже.';
  if (status === 404) return 'Задание не найдено. Запустите обработку ещё раз.';
  return `Ошибка сервера (HTTP ${status}).`;
}

async function request<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      method: init.method ?? 'GET',
      headers: init.body === undefined ? { Accept: 'application/json' } : { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
      signal: controller.signal,
    });
    const text = await response.text();
    let body: unknown = null;
    if (text) {
      try { body = JSON.parse(text); } catch { body = null; }
    }
    if (!response.ok) throw new Error(describeError(response.status, body));
    if (body === null || typeof body !== 'object') throw new Error('Сервер вернул неожиданный ответ.');
    return body as T;
  } catch (error) {
    if (error instanceof Error && error.name !== 'AbortError' && !(error instanceof TypeError)) throw error;
    if (controller.signal.aborted) throw new Error('Сервер не ответил за 30 секунд. Попробуйте ещё раз.');
    throw new Error('Нет связи с сервером классификации. Проверьте, что API запущен.');
  } finally {
    clearTimeout(timer);
  }
}

function toResult(item: ApiClassify): Result {
  const notFound = item.status === 'not_found';
  return {
    confidenceKind: item.confidence_kind ?? 'uncalibrated',
    alternatives: item.alternatives ?? [],
    name: item.query,
    code: notFound ? null : item.code,
    title: notFound ? null : item.title,
    confidence: notFound ? null : item.confidence,
    status: item.status,
    clauseRef: item.clause_ref,
    url: item.clause_url,
  };
}

export const api: ClassifierApi = {
  async classify(name) {
    const body = await request<ApiClassify>('/classify', { method: 'POST', body: { name } });
    return toResult(body);
  },
  async startBatch(names) {
    const body = await request<ApiAccepted>('/classify/batch', { method: 'POST', body: { items: names } });
    return { job_id: body.job_id };
  },
  async getBatch(id) {
    const body = await request<ApiBatch>(`/classify/batch/${encodeURIComponent(id)}`);
    return {
      job_id: body.job_id,
      status: body.status,
      processed: body.completed,
      total: body.total,
      results: (body.results ?? []).map(toResult),
      error: body.error,
    };
  },
};
