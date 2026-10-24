import type { Result } from './api';
import { STATUS_LABEL } from './status.ts';

// ExcelJS не нужен при одиночном поиске: загружаем только во время экспорта.
function absoluteUrl(url: string | null): string {
  if (!url) return '';
  try { return new URL(url, window.location.origin).href; } catch { return ''; }
}

export async function exportResults(results: Result[]) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook();
  const sheet = book.addWorksheet('Результаты');
  sheet.addRow(['Товар', 'Код ОКПД2', 'Наименование', 'Статус', 'Уверенность', 'Исходный скор', 'Пункт перечня', 'Ссылка', 'Оценка пользователя', 'Исправленный код', 'Проверяющая ступень', 'Пометка ступени']);
  results.forEach(r => sheet.addRow([
    r.name, r.code ?? '', r.title ?? '', STATUS_LABEL[r.status],
    r.confidence ?? '', r.confidenceRaw ?? '', r.clauseRef ?? '', absoluteUrl(r.url),
    r.userVerdict ?? '', r.correctedCode ?? '', r.verifier ?? '', r.verifierNote ?? '',
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
    // CSV quoting alone does not stop spreadsheet programs from evaluating formulas.
    if (typeof value === 'string' && /^(?:\s*[=+@-]|[\t\r\n])/.test(s)) s = `'${s}`;
    return /[";\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  // verifier/verifier_note едут последними: старые потребители читают первые
  // десять колонок по именам, а проверяющая ступень должна быть видна в выгрузке.
  const lines = [['query', 'code', 'title', 'status', 'confidence', 'confidence_raw', 'clause_ref', 'clause_url', 'user_verdict', 'corrected_code', 'verifier', 'verifier_note'].join(';')];
  results.forEach(r => lines.push([
    esc(r.name), esc(r.code), esc(r.title), esc(r.status), esc(r.confidence),
    esc(r.confidenceRaw), esc(r.clauseRef), esc(absoluteUrl(r.url)),
    esc(r.userVerdict ?? null), esc(r.correctedCode ?? null),
    esc(r.verifier ?? null), esc(r.verifierNote ?? null),
  ].join(';')));
  download(new Blob([`\uFEFF${lines.join('\r\n')}\r\n`], { type: 'text/csv;charset=utf-8' }), 'classification-results.csv');
}
