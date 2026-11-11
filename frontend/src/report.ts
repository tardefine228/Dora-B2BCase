import type {
  ProcurementReport,
  ProcurementReportItem,
  ReportBucket,
  ReportReasonCode,
  Result,
} from './api';

export const BUCKET_LABEL: Record<ReportBucket, string> = {
  accepted: 'Принято',
  needs_review: 'Требуют проверки',
  not_found: 'Не найдено',
};

export const REASON_LABEL: Record<ReportReasonCode, string> = {
  multiple_codes: 'Несколько кодов',
  code_exception: 'Исключение / роль кода',
  ambiguous_name: 'Неоднозначное название',
  catalog_review_required: 'Сверка каталога',
  code_missing: 'Код не назначен',
  low_confidence: 'Низкая уверенность',
  not_found: 'Не найдено в перечне',
};

export function reportItem(
  report: ProcurementReport | null,
  result: Result,
  index: number,
): ProcurementReportItem | null {
  if (!report) return null;
  const id = result.itemId ?? index;
  return report.items.find(item => item.itemId === id) ?? null;
}

export function filterReportResults(
  results: Result[],
  report: ProcurementReport | null,
  bucket: ReportBucket | 'all',
  reason: ReportReasonCode | 'all',
): Result[] {
  if (bucket === 'all' && reason === 'all') return results;
  return results.filter((result, index) => {
    const item = reportItem(report, result, index);
    if (!item) return false;
    return (bucket === 'all' || item.bucket === bucket)
      && (reason === 'all' || item.reasons.some(entry => entry.code === reason));
  });
}
