import Papa from 'papaparse';
import ExcelJS from 'exceljs';
import type { Result } from './api';
export type Table = { headers: string[]; rows: string[][] };
export async function readTable(file: File): Promise<Table> {
  if (file.size > 5 * 1024 * 1024) throw new Error('Файл слишком большой. Максимум 5 МБ.');
  const ext = file.name.split('.').pop()?.toLowerCase();
  let rows: string[][];
  if (ext === 'csv') {
    const text = (await file.text()).replace(/^\uFEFF/, '');
    if (text.includes('\uFFFD')) throw new Error('Сохраните CSV в кодировке UTF-8 и загрузите снова.');
    const parsed = Papa.parse<string[]>(text, { skipEmptyLines: 'greedy' });
    if (parsed.errors.some(e => e.type === 'Quotes')) throw new Error('В CSV нарушено оформление кавычек.');
    rows = parsed.data;
  } else if (ext === 'xlsx') {
    const book = new ExcelJS.Workbook();
    await book.xlsx.load(await file.arrayBuffer());
    const sheet = book.worksheets[0];
    if (!sheet) throw new Error('В Excel нет листов.');
    rows = [];
    sheet.eachRow(row => {
      const cells: string[] = [];
      row.eachCell({ includeEmpty: true }, (cell, col) => { cells[col - 1] = cell.text; });
      rows.push(cells);
    });
  } else throw new Error('Выберите .xlsx или .csv. Старый .xls сохраните как .xlsx.');
  rows = rows.filter(row => row.some(value => String(value ?? '').trim()));
  if (rows.length < 2) throw new Error('Нужны строка заголовков и хотя бы один товар.');
  if (rows.length > 101) throw new Error('Максимум 100 строк товаров. Разделите файл на несколько частей.');
  const width = Math.max(...rows.map(row => row.length));
  if (width > 100) throw new Error('В файле слишком много столбцов (максимум 100).');
  const headers = Array.from({ length: width }, (_, i) => rows[0][i]?.trim() || `Столбец ${i + 1}`);
  return { headers, rows: rows.slice(1) };
}
export function namesFrom(table: Table, column: number): string[] {
  return table.rows.map((row, index) => {
    const name = (row[column] ?? '').trim();
    if (!name) throw new Error(`В строке данных ${index + 1} нет названия товара.`);
    if (name.length > 500) throw new Error(`В строке данных ${index + 1} название длиннее 500 символов.`);
    return name;
  });
}
const STATUS_TEXT = { ok: 'Уверенный результат', review: 'Требует проверки', not_found: 'Не найдено' } as const;

// Относительные ссылки `/source/...` делаем абсолютными, чтобы они открывались из файла.
function absoluteUrl(url: string | null): string {
  if (!url) return '';
  try { return new URL(url, window.location.origin).href; } catch { return url; }
}

export async function exportResults(results: Result[]) {
  const book = new ExcelJS.Workbook();
  const sheet = book.addWorksheet('Результаты');
  sheet.addRow(['Товар', 'Код ОКПД2', 'Наименование', 'Статус', 'Уверенность', 'Пункт перечня', 'Ссылка']);
  results.forEach(r => sheet.addRow([
    r.name,
    r.code ?? '',
    r.title ?? '',
    STATUS_TEXT[r.status],
    r.confidence ?? '',
    r.clauseRef ?? '',
    absoluteUrl(r.url),
  ]));
  sheet.getRow(1).font = { bold: true };
  sheet.columns.forEach(col => { col.width = 32; });
  sheet.getColumn(5).numFmt = '0%';
  const buffer = await book.xlsx.writeBuffer();
  const blob = new Blob([buffer as BlobPart], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'classification-results.xlsx'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Тот же формат, что отдаёт серверный эндпоинт /classify/batch/{job_id}.csv:
// «;», UTF-8 с BOM, CRLF — Excel русской локали открывает без мастеров импорта.
export function exportResultsCsv(results: Result[]) {
  const esc = (value: string | number | null) => {
    const s = value === null || value === undefined ? '' : String(value);
    return /[";\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [['query', 'code', 'title', 'status', 'confidence', 'clause_ref', 'clause_url'].join(';')];
  results.forEach(r => lines.push([
    esc(r.name), esc(r.code), esc(r.title), esc(r.status), esc(r.confidence), esc(r.clauseRef), esc(absoluteUrl(r.url)),
  ].join(';')));
  const blob = new Blob([`\uFEFF${lines.join('\r\n')}\r\n`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'classification-results.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
