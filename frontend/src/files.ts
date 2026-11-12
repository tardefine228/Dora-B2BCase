import type { ProcurementReport, Result } from './api';
import { BUCKET_LABEL, REASON_LABEL, reportItem } from './report.ts';
import { STATUS_LABEL } from './status.ts';
import { procurementVerdict, PROCUREMENT_SCOPE } from './procurement.ts';
import { clarificationOptionExport } from './clarification.ts';

const procurementColumns = (r: Result) => {
  const v = procurementVerdict(r);
  return [v.title, [v.caution, v.explanation, PROCUREMENT_SCOPE].filter(Boolean).join(' ')];
};

// ExcelJS не нужен при одиночном поиске: загружаем только во время экспорта.
function absoluteUrl(url: string | null): string {
  if (!url) return '';
  try { return new URL(url, window.location.origin).href; } catch { return ''; }
}

// Номер в первой колонке — тот же счётчик 1…N, что в колонке «№» на экране.
export async function exportResults(results: Result[], report: ProcurementReport | null = null) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook();
  const headers = ['№', 'Исходный №', 'Товар', 'Код ОКПД2', 'Наименование', 'Статус', 'Уверенность', 'Исходный скор', 'Пункт перечня', 'Ссылка', 'Оценка пользователя', 'Исправленный код', 'Проверяющая ступень', 'Пометка ступени', 'Условия закупки', 'Пояснение условий', 'Источник решения', 'Выбранный вариант', 'Вопрос уточнения', 'Варианты уточнения', 'Исходный код модели', 'Раздел отчёта', 'Ранг риска', 'Причины проверки'];
  const addRows = (sheet: import('exceljs').Worksheet, rows: Result[]) => {
    sheet.addRow(headers);
    rows.forEach((r, index) => {
      const item = reportItem(report, r, r.itemId ?? index);
      sheet.addRow([
        index + 1, (r.itemId ?? index) + 1,
        r.name, r.code ?? '', r.title ?? '', STATUS_LABEL[r.status],
        r.confidence ?? '', r.confidenceRaw ?? '', r.clauseRef ?? '', absoluteUrl(r.url),
        r.userVerdict ?? '', r.correctedCode ?? '', r.verifier ?? '', r.verifierNote ?? '', ...procurementColumns(r),
        r.decisionSource, r.resolution?.optionId ?? '', r.clarification?.question ?? '',
        r.clarification?.options.map(option =>
          clarificationOptionExport(option, r.clarification?.options ?? [], false)).join('\n') ?? '',
        r.resolution?.original.code ?? r.clarification?.original.code ?? '',
        item ? BUCKET_LABEL[item.bucket] : '', item?.riskRank ?? '',
        item?.reasons.map(reason => `${REASON_LABEL[reason.code]}: ${reason.message}`).join('\n') ?? '',
      ]);
    });
    sheet.getRow(1).font = { bold: true };
    sheet.columns.forEach(col => { col.width = 32; });
    sheet.getColumn(1).width = 7;
    sheet.getColumn(2).width = 12;
    sheet.getColumn(7).numFmt = '0%';
    sheet.getColumn(8).numFmt = '0%';
  };
  addRows(book.addWorksheet('Результаты'), results);
  if (report) {
    const summary = book.addWorksheet('Сводка');
    summary.addRow(['Итоги закупки', 'Количество']);
    summary.addRow(['Всего', report.summary.total]);
    summary.addRow(['Принято', report.summary.accepted]);
    summary.addRow(['Автоматически', report.summary.autoAccepted]);
    summary.addRow(['Специалистом', report.summary.specialistResolved]);
    summary.addRow(['Требуют проверки', report.summary.needsReview]);
    summary.addRow(['Не найдено', report.summary.notFound]);
    summary.addRow([]);
    summary.addRow(['Причина', 'Позиций']);
    Object.entries(report.summary.reasonCounts).forEach(([code, count]) =>
      summary.addRow([REASON_LABEL[code as keyof typeof REASON_LABEL], count]));
    summary.getRow(1).font = { bold: true };
    summary.columns = [{ width: 36 }, { width: 16 }];
    const review = results.map((result, index) => ({ result, index, item: reportItem(report, result, result.itemId ?? index) }))
      .filter(row => row.item?.bucket === 'needs_review')
      .sort((a, b) => (b.item?.riskRank ?? 0) - (a.item?.riskRank ?? 0) || a.index - b.index)
      .map(row => row.result);
    addRows(book.addWorksheet('Требуют проверки'), review);
  }
  const buffer = await book.xlsx.writeBuffer();
  const blob = new Blob([buffer as BlobPart], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
  download(blob, 'classification-results.xlsx');
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a);
  a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function exportResultsCsv(results: Result[], report: ProcurementReport | null = null) {
  const esc = (value: string | number | null) => {
    let s = value === null || value === undefined ? '' : String(value);
    // CSV quoting alone does not stop spreadsheet programs from evaluating formulas.
    if (typeof value === 'string' && /^(?:\s*[=+@-]|[\t\r\n])/.test(s)) s = `'${s}`;
    return /[";\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  // Preserve the first 13 columns; append the same procurement explanation shown in the UI.
  const lines = [['query', 'code', 'title', 'status', 'confidence', 'confidence_raw', 'clause_ref', 'clause_url', 'user_verdict', 'corrected_code', 'verifier', 'verifier_note', 'row', 'procurement_regime', 'procurement_explanation', 'decision_source', 'selected_option_id', 'clarification_question', 'clarification_options', 'original_code', 'original_status', 'source_row', 'report_bucket', 'risk_rank', 'report_reasons'].join(';')];
  results.forEach((r, index) => {
    const item = reportItem(report, r, r.itemId ?? index);
    lines.push([
    esc(r.name), esc(r.code), esc(r.title), esc(r.status), esc(r.confidence),
    esc(r.confidenceRaw), esc(r.clauseRef), esc(absoluteUrl(r.url)),
    esc(r.userVerdict ?? null), esc(r.correctedCode ?? null),
    esc(r.verifier ?? null), esc(r.verifierNote ?? null),
    esc(index + 1), ...procurementColumns(r).map(esc),
    esc(r.decisionSource), esc(r.resolution?.optionId ?? null),
    esc(r.clarification?.question ?? null),
    esc(r.clarification?.options.map(option =>
      clarificationOptionExport(option, r.clarification?.options ?? [])).join(' | ') ?? null),
    esc(r.resolution?.original.code ?? r.clarification?.original.code ?? null),
    esc(r.resolution?.original.status ?? r.clarification?.original.status ?? null),
    esc((r.itemId ?? index) + 1), esc(item?.bucket ?? null), esc(item?.riskRank ?? null),
    esc(item?.reasons.map(reason => `${reason.code}: ${reason.message}`).join(' | ') ?? null),
  ].join(';'));
  });
  download(new Blob([`\uFEFF${lines.join('\r\n')}\r\n`], { type: 'text/csv;charset=utf-8' }), 'classification-results.csv');
}
