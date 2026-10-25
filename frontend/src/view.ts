import type { Result, UploadAccepted } from './api';

// Текущий просмотр (результаты последней проверки) переживает только возврат на
// страницу по «Назад»: пользователь ушёл по ссылке «Открыть пункт постановления»
// в той же вкладке и вернулся. Всё остальное — новый запуск сайта, перезагрузка,
// восстановленная после перезапуска браузера вкладка — должно начинаться с
// пустой страницы: показывать разобранный файл, который в этой сессии не
// запускали, нельзя (это и был баг: sessionStorage переживает перезапуск).
export const VIEW_KEY = 'dora:current-view:v1';

export type Progress = { processed: number; total: number; state: 'running' | 'done' | 'failed' };

export type StoredView = {
  name?: string;
  result?: Result | null;
  filename?: string;
  fileInfo?: UploadAccepted | null;
  progress?: Progress | null;
  results?: Result[];
};

type ViewStorage = Pick<Storage, 'getItem' | 'removeItem'>;

/** Тип загрузки страницы: `navigate` | `reload` | `back_forward` | `prerender`. */
export function navigationType(entries?: unknown): string {
  const list = entries ?? (typeof performance !== 'undefined' && performance.getEntriesByType
    ? performance.getEntriesByType('navigation') : undefined);
  const first = Array.isArray(list) ? list[0] as { type?: unknown } | undefined : undefined;
  return typeof first?.type === 'string' ? first.type : 'navigate';
}

/**
 * Возвращается ли пользователь на уже открытую страницу.
 *
 * Только `back_forward`: это единственный случай, ради которого просмотр вообще
 * сохраняется. `reload` и `navigate` — новый запуск сайта, и прошлые результаты
 * в нём не показываются.
 */
export function shouldRestoreView(type: string): boolean {
  return type === 'back_forward';
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const STATUSES = new Set(['ok', 'review', 'not_found']);

/** Результат из хранилища в том виде, в котором его можно рендерить. */
function sanitizeResult(value: unknown): Result | null {
  if (!isObject(value) || typeof value.name !== 'string') return null;
  if (typeof value.status !== 'string' || !STATUSES.has(value.status)) return null;
  const alternatives = Array.isArray(value.alternatives)
    ? value.alternatives.filter(isObject) as Result['alternatives']
    : [];
  return { ...value, alternatives } as Result;
}

/** Сведения о разборе файла: UI читает их списком, поэтому массив обязателен. */
function sanitizeFileInfo(value: unknown): UploadAccepted | undefined {
  if (!isObject(value) || typeof value.total !== 'number' || !Number.isFinite(value.total)) return undefined;
  const number = (key: string) => (typeof value[key] === 'number' ? value[key] as number : 0);
  return {
    ...value,
    total: value.total,
    total_found: number('total_found'),
    duplicates: number('duplicates'),
    skipped: number('skipped'),
    truncated: Boolean(value.truncated),
    source_column: typeof value.source_column === 'string' ? value.source_column : null,
    warnings: Array.isArray(value.warnings)
      ? value.warnings.filter((warning): warning is string => typeof warning === 'string')
      : [],
  } as UploadAccepted;
}

function sanitizeProgress(value: unknown): Progress | undefined {
  if (!isObject(value)) return undefined;
  const { processed, total, state } = value;
  if (typeof processed !== 'number' || typeof total !== 'number' || total <= 0) return undefined;
  if (state !== 'running' && state !== 'done' && state !== 'failed') return undefined;
  return { processed, total, state };
}

/** Оставляет только те поля, которые можно показать, не доверяя хранилищу. */
export function sanitizeView(value: unknown): StoredView {
  if (!isObject(value)) return {};
  const view: StoredView = {};
  if (typeof value.name === 'string') view.name = value.name;
  if (typeof value.filename === 'string') view.filename = value.filename;
  const fileInfo = sanitizeFileInfo(value.fileInfo);
  if (fileInfo) view.fileInfo = fileInfo;
  const progress = sanitizeProgress(value.progress);
  if (progress) view.progress = progress;
  if (Array.isArray(value.results)) {
    view.results = value.results
      .map(sanitizeResult)
      .filter((result): result is Result => result !== null);
  }
  const single = sanitizeResult(value.result);
  if (single) view.result = single;
  return view;
}

/**
 * Прошлый просмотр — или `null`, если показывать его не нужно.
 *
 * Запись удаляется, когда восстанавливать её не следует: иначе она доживёт до
 * следующего возврата по «Назад» и снова покажет чужой сессии файл.
 */
export function readStoredView(storage: ViewStorage, type: string): StoredView | null {
  const raw = storage.getItem(VIEW_KEY);
  if (!raw) return null;
  if (!shouldRestoreView(type)) {
    storage.removeItem(VIEW_KEY);
    return null;
  }
  try {
    return sanitizeView(JSON.parse(raw));
  } catch {
    storage.removeItem(VIEW_KEY);
    return null;
  }
}

/**
 * Что делать с заданием, найденным при открытии страницы.
 *
 * Задание хранится в localStorage, потому что обработка переживает закрытие
 * вкладки. Возобновлять его надо только если оно ещё идёт: завершённое задание —
 * это чужой (для этой сессии) результат, и показывать его нельзя.
 */
export type JobMarker = { id: string; total: number; filename?: string };

export function parseJobMarker(raw: string | null): JobMarker | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw);
    if (!isObject(value)) return null;
    const id = value.id;
    const total = value.total;
    if (typeof id !== 'string' || !id) return null;
    if (typeof total !== 'number' || !Number.isInteger(total) || total <= 0 || total > 100) return null;
    const marker: JobMarker = { id, total };
    if (typeof value.filename === 'string') marker.filename = value.filename;
    return marker;
  } catch {
    return null;
  }
}

export function isRunningJob(status: string): boolean {
  return status === 'queued' || status === 'running';
}
