import test from 'node:test';
import assert from 'node:assert/strict';
import { api } from '../src/api.ts';
import { filterReportResults } from '../src/report.ts';
import { orderResults } from '../src/rows.ts';
import { exportResults, exportResultsCsv } from '../src/files.ts';

const report = {
  summary: { total: 3, accepted: 1, autoAccepted: 1, specialistResolved: 0, needsReview: 1, notFound: 1, reasonCounts: { ambiguous_name: 1, not_found: 1 } },
  items: [
    { itemId: 0, bucket: 'accepted', riskRank: 0, reasons: [] },
    { itemId: 1, bucket: 'needs_review', riskRank: 90, reasons: [{ code: 'ambiguous_name', message: 'Выберите вариант.' }] },
    { itemId: 2, bucket: 'not_found', riskRank: 60, reasons: [{ code: 'not_found', message: 'Не найдено.' }] },
  ],
};
const rows = [
  { itemId: 0, name: 'Бумага', code: '17.12', title: 'Бумага', status: 'ok', confidence: .9, confidenceRaw: .8, alternatives: [], decisionSource: 'model', attentionReasons: [] },
  { itemId: 1, name: 'Автомобиль', code: null, title: null, status: 'needs_clarification', confidence: null, confidenceRaw: 0, alternatives: [], decisionSource: 'model', attentionReasons: [] },
  { itemId: 2, name: 'Неизвестный товар', code: null, title: null, status: 'not_found', confidence: 0, confidenceRaw: 0, alternatives: [], decisionSource: 'model', attentionReasons: [] },
];

test('risk sort and report filters use server item ids and reasons', () => {
  assert.deepEqual(orderResults(rows, 'risk', report).map(row => row.itemId), [1, 2, 0]);
  assert.deepEqual(filterReportResults(rows, report, 'needs_review', 'all').map(row => row.itemId), [1]);
  assert.deepEqual(filterReportResults(rows, report, 'all', 'not_found').map(row => row.itemId), [2]);
  assert.deepEqual(filterReportResults(rows, report, 'accepted', 'not_found'), []);
  assert.equal(rows[0].itemId, 0, 'input order is not mutated');
});

test('batch adapter preserves backend summary instead of recalculating it', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({
    job_id: 'report', status: 'completed', total: 3, completed: 3, error: null,
    results: [],
    report: {
      summary: { total: 3, accepted: 2, auto_accepted: 1, specialist_resolved: 1, needs_review: 1, not_found: 0, reason_counts: { low_confidence: 1 } },
      items: [{ item_id: 4, bucket: 'needs_review', risk_rank: 70, reasons: [{ code: 'low_confidence', message: 'Проверить.' }] }],
    },
  }), { status: 200 }));
  const job = await api.getBatch('report');
  assert.equal(job.report.summary.specialistResolved, 1);
  assert.equal(job.report.summary.reasonCounts.low_confidence, 1);
  assert.deepEqual(job.report.items[0], { itemId: 4, bucket: 'needs_review', riskRank: 70, reasons: [{ code: 'low_confidence', message: 'Проверить.' }] });
});

test('XLSX and CSV exports contain the same server report and risk order', async t => {
  const blobs = [];
  globalThis.window = { location: { origin: 'http://test' } };
  globalThis.document = {
    body: { appendChild() {} },
    createElement() { return { click() {}, remove() {}, set href(value) { this.url = value; } }; },
  };
  t.after(() => { delete globalThis.window; delete globalThis.document; });
  t.mock.method(URL, 'createObjectURL', blob => { blobs.push(blob); return 'blob:test'; });
  t.mock.method(URL, 'revokeObjectURL', () => {});

  await exportResults(orderResults(rows, 'risk', report), report);
  exportResultsCsv(orderResults(rows, 'risk', report), report);
  assert.equal(blobs.length, 2);
  const { default: ExcelJS } = await import('exceljs');
  const book = new ExcelJS.Workbook();
  await book.xlsx.load(Buffer.from(await blobs[0].arrayBuffer()));
  assert.deepEqual(book.worksheets.map(sheet => sheet.name), ['Результаты', 'Сводка', 'Требуют проверки']);
  assert.equal(book.getWorksheet('Сводка').getCell('B3').value, 1);
  const review = book.getWorksheet('Требуют проверки');
  assert.equal(review.getCell('C2').value, 'Автомобиль');
  assert.equal(review.getCell('W2').value, 90);
  assert.match(review.getCell('X2').value, /Выберите вариант/);
  const csv = await blobs[1].text();
  assert.match(csv, /report_bucket;risk_rank;report_reasons/);
  assert.match(csv, /needs_review;90;ambiguous_name: Выберите вариант/);
});
