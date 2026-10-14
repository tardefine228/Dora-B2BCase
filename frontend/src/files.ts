import Papa from 'papaparse';
import type { Result } from './api';
type Sheet = { name: string; raw: string[][] };
export type Table = { headers: string[]; rows: string[][]; raw: string[][]; hasHeader: boolean; sheets: Sheet[]; sheetIndex: number; column: number };
const headerScore = (s: string) => /^(?:наименование(?: товара| продукции| позиции| объекта закупки)?|название(?: товара| позиции)?|товар|номенклатура|product_name|product|name)$/i.test(s.trim().replace(/\s+/g, ' ')) ? 2 : /^(?:описание(?: товара)?|description)$/i.test(s.trim()) ? 1 : 0;
export function makeTable(raw: string[][], sheets: Sheet[] = [], sheetIndex = 0, hasHeader?: boolean): Table {
  raw = raw.filter(r => r.some(v => String(v ?? '').trim())).map(r => r.map(v => String(v ?? '').trim()));
  if (!raw.length) throw new Error('В файле нет данных.');
  const width = raw.reduce((max, row) => Math.max(max, row.length), 0);
  if (width > 100) throw new Error('Максимум 100 столбцов.');
  const headerIndex = raw.slice(0, 10).findIndex(r => r.some(v => headerScore(v) > 0));
  if (hasHeader === undefined && headerIndex > 0) raw = raw.slice(headerIndex);
  const detected = hasHeader ?? (headerIndex >= 0);
  const headers = Array.from({ length: width }, (_, i) => detected ? raw[0][i] || `Столбец ${i + 1}` : `Столбец ${i + 1}`);
  const rows = detected ? raw.slice(1) : raw;
  const scores: number[] = headers.map(headerScore);
  const textScores = Array.from({ length: width }, (_, i) => rows.slice(0, 20).reduce((score, row) => score + (/[a-zа-яё]{2}/i.test(row[i] ?? '') ? 1 : 0), 0));
  const column = Math.max(...scores) > 0 ? scores.indexOf(Math.max(...scores)) : Math.max(0, textScores.indexOf(Math.max(...textScores)));
  return { headers, rows, raw, sheets, sheetIndex, hasHeader: detected, column };
}
export function switchSheet(table: Table, sheetIndex: number): Table {
  return makeTable(table.sheets[sheetIndex].raw, table.sheets, sheetIndex);
}
export function switchHeader(table: Table, hasHeader: boolean): Table {
  return makeTable(table.raw, table.sheets, table.sheetIndex, hasHeader);
}
export async function readTable(file: File): Promise<Table> {
  if (file.size > 5 * 1024 * 1024) throw new Error('Максимальный размер файла — 5 МБ.');
  const ext = file.name.split('.').pop()?.toLowerCase();
  const sheets: Sheet[] = [];
  if (ext === 'csv') {
    let text = new TextDecoder('utf-8', { fatal: false }).decode(await file.arrayBuffer()).replace(/^\uFEFF/, '');
    if (text.includes('\uFFFD')) text = new TextDecoder('windows-1251').decode(await file.arrayBuffer());
    const parsed = Papa.parse<string[]>(text, { skipEmptyLines: 'greedy' });
    if (parsed.errors.some(e => e.type === 'Quotes')) throw new Error('В CSV нарушено оформление кавычек.');
    sheets.push({ name: file.name, raw: parsed.data });
  } else if (ext === 'xlsx') {
    const { default: ExcelJS } = await import('exceljs');
    const book = new ExcelJS.Workbook();
    await book.xlsx.load(await file.arrayBuffer());
    for (const sheet of book.worksheets) {
      const raw: string[][] = [];
      sheet.eachRow(row => {
        const cells: string[] = [];
        row.eachCell({ includeEmpty: true }, (cell, col) => { cells[col - 1] = cell.text; });
        raw.push(cells);
      });
      if (raw.length) sheets.push({ name: sheet.name, raw });
    }
  } else throw new Error('Выберите .xlsx или .csv. Старый .xls сохраните как .xlsx.');
  if (!sheets.length) throw new Error('В файле нет непустых листов.');
  return makeTable(sheets[0].raw, sheets);
}
export function namesFrom(table: Table, column: number): string[] {
  const names = table.rows.map(row => (row[column] ?? '').trim()).filter(Boolean);
  if (!names.length) throw new Error('В выбранном столбце нет названий.');
  if (names.length > 100) throw new Error('Найдено больше 100 товаров. Разделите список на несколько файлов.');
  const long = names.findIndex(n => n.length > 500);
  if (long >= 0) throw new Error(`Название товара №${long + 1} длиннее 500 символов.`);
  return names;
}
const STATUS_TEXT = { ok: 'Уверенный результат', review: 'Требует проверки', not_found: 'Не найдено' } as const;
export async function exportResults(results: Result[]) {
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook();
  const sheet = book.addWorksheet('Результаты');
  sheet.addRow(['Товар', 'Код ОКПД2', 'Наименование', 'Статус', 'Оценка', 'Тип оценки', 'Пункт перечня', 'Ссылка']);
  results.forEach(r => {
    const rawUrl = r.url?.startsWith('/source/') ? `${r.url}?format=html` : r.url;
    const url = rawUrl && /^(?:\/source\/|https?:\/\/)/i.test(rawUrl) ? new URL(rawUrl, window.location.origin).href : '';
    sheet.addRow([r.name, r.code ?? '', r.title ?? '', STATUS_TEXT[r.status], r.confidence ?? '', r.confidenceKind === 'calibrated' ? 'Уверенность модели' : r.confidenceKind === 'exact' ? 'Точное совпадение кода' : 'Сходство, не вероятность', r.clauseRef ?? '', url]);
  });
  sheet.getRow(1).font = { bold: true };
  sheet.columns.forEach(col => { col.width = 32; });
  sheet.getColumn(5).numFmt = '0%';
  const buffer = await book.xlsx.writeBuffer();
  const url = URL.createObjectURL(new Blob([buffer as BlobPart], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
  const a = document.createElement('a'); a.href = url; a.download = 'classification-results.xlsx'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
