import type { Result } from './api';

// ExcelJS не нужен при одиночном поиске: загружаем только во время экспорта.
function absoluteUrl(url: string | null): string {
  if (!url) return '';
  try { return new URL(url, window.location.origin).href; } catch { return ''; }
}
const STATUS_TEXT = {
  ok: 'Уверенный результат', review: 'Требует проверки', not_found: 'Не найдено',
} as const;

export async function exportResults(results: Result[]) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook();
  const sheet = book.addWorksheet('Результаты');
  sheet.addRow(['Товар', 'Код ОКПД2', 'Наименование', 'Статус', 'Уверенность', 'Исходный скор', 'Пункт перечня', 'Ссылка', 'Оценка пользователя', 'Исправленный код']);
  results.forEach(r => sheet.addRow([
    r.name, r.code ?? '', r.title ?? '', STATUS_TEXT[r.status],
    r.confidence ?? '', r.confidenceRaw ?? '', r.clauseRef ?? '', absoluteUrl(r.url),
    r.userVerdict ?? '', r.correctedCode ?? '',
  ]));
  sheet.getRow(1).font = { bold: true };
  sheet.columns.forEach(col => { col.width = 32; });
  sheet.getColumn(5).numFmt = '0%';
  sheet.getColumn(6).numFmt = '0%';
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

export function exportResultsCsv(results: Result[]) {
  const esc = (value: string | number | null) => {
    let s = value === null || value === undefined ? '' : String(value);
    // Quoting a CSV field does not prevent spreadsheet formula evaluation.
    if (typeof value === 'string' && /^(?:\s*[=+@-]|[\t\r\n])/.test(s)) s = `'${s}`;
    return /[";\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [['query', 'code', 'title', 'status', 'confidence', 'confidence_raw', 'clause_ref', 'clause_url', 'user_verdict', 'corrected_code'].join(';')];
  results.forEach(r => lines.push([
    esc(r.name), esc(r.code), esc(r.title), esc(r.status), esc(r.confidence),
    esc(r.confidenceRaw), esc(r.clauseRef), esc(absoluteUrl(r.url)),
    esc(r.userVerdict ?? null), esc(r.correctedCode ?? null),
  ].join(';')));
  download(new Blob([`\uFEFF${lines.join('\r\n')}\r\n`], { type: 'text/csv;charset=utf-8' }), 'classification-results.csv');
}
