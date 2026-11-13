import type { JobStatus, UploadAccepted } from './api';

export const FILE_HISTORY_KEY = 'dora:file-history:v1';
export const FILE_HISTORY_LIMIT = 3;

/** Only report metadata is stored locally; product names and rows stay on the server. */
export type FileHistoryEntry = {
  jobId: string;
  filename: string;
  total: number;
  createdAt: string;
  status: JobStatus;
  fileInfo: UploadAccepted | null;
};

const STATUSES = new Set<JobStatus>(['queued', 'running', 'completed', 'failed']);

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function sanitizeFileInfo(value: unknown): UploadAccepted | null {
  if (!isObject(value) || typeof value.total !== 'number' || !Number.isFinite(value.total)) return null;
  const number = (key: string) => typeof value[key] === 'number' ? value[key] as number : 0;
  return {
    ...value,
    job_id: typeof value.job_id === 'string' ? value.job_id : '',
    filename: typeof value.filename === 'string' ? value.filename : '',
    format: typeof value.format === 'string' ? value.format : '',
    total: value.total,
    total_found: number('total_found'), duplicates: number('duplicates'), skipped: number('skipped'),
    truncated: Boolean(value.truncated),
    source_column: typeof value.source_column === 'string' ? value.source_column : null,
    warnings: Array.isArray(value.warnings)
      ? value.warnings.filter((item): item is string => typeof item === 'string') : [],
  } as UploadAccepted;
}

export function parseFileHistory(raw: string | null): FileHistoryEntry[] {
  if (!raw) return [];
  try {
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    return value.filter(isObject).flatMap(item => {
      if (typeof item.jobId !== 'string' || !item.jobId || typeof item.filename !== 'string'
        || !Number.isInteger(item.total) || (item.total as number) <= 0 || (item.total as number) > 100
        || typeof item.createdAt !== 'string' || !STATUSES.has(item.status as JobStatus)) return [];
      return [{
        jobId: item.jobId, filename: item.filename, total: item.total as number,
        createdAt: item.createdAt, status: item.status as JobStatus,
        fileInfo: sanitizeFileInfo(item.fileInfo),
      }];
    }).slice(0, FILE_HISTORY_LIMIT);
  } catch { return []; }
}

export function addFileHistory(
  entries: readonly FileHistoryEntry[], entry: FileHistoryEntry,
): FileHistoryEntry[] {
  return [entry, ...entries.filter(item => item.jobId !== entry.jobId)].slice(0, FILE_HISTORY_LIMIT);
}

export function updateFileHistoryStatus(
  entries: readonly FileHistoryEntry[], jobId: string, status: JobStatus,
): FileHistoryEntry[] {
  return entries.map(item => item.jobId === jobId ? { ...item, status } : item);
}
