import assert from 'node:assert/strict';
import { test } from 'node:test';
import { numberResults, orderResults } from '../src/rows.ts';

const item = (name, confidence) => ({
  name, confidence, code: null, title: null, status: 'review',
  confidenceRaw: confidence, clauseRef: null, url: null, alternatives: [],
});
// Порядок файла: 1 «Ноутбук», 2 «Бумага», 3 «Стол».
const file = [item('Ноутбук', 0.4), item('Бумага', 0.9), item('Стол', 0.6)];

test('numbering is the position in the uploaded file', () => {
  assert.deepEqual(numberResults(file).map(row => row.position), [1, 2, 3]);
  assert.deepEqual(numberResults(file).map(row => row.result.name), ['Ноутбук', 'Бумага', 'Стол']);
});

test('changing the sort never changes a row number', () => {
  const rows = numberResults(file);
  const expected = new Map(rows.map(row => [row.result.name, row.position]));

  for (const sort of ['original', 'name', 'confidence']) {
    const ordered = orderResults(rows, sort);
    assert.equal(ordered.length, rows.length);
    // Набор номеров тот же, и каждый номер по-прежнему принадлежит своему товару.
    assert.deepEqual([...ordered.map(row => row.position)].sort(), [1, 2, 3]);
    for (const row of ordered) assert.equal(row.position, expected.get(row.result.name));
  }
});

test('sorting reorders rows, not numbers: the visible column is not 1,2,3', () => {
  const rows = numberResults(file);
  // По уверенности: «Бумага» (0.9, № 2), «Стол» (0.6, № 3), «Ноутбук» (0.4, № 1).
  assert.deepEqual(orderResults(rows, 'confidence').map(row => [row.position, row.result.name]),
    [[2, 'Бумага'], [3, 'Стол'], [1, 'Ноутбук']]);
  // По названию: «Бумага» (2), «Ноутбук» (1), «Стол» (3) — кириллица, ru-локаль.
  assert.deepEqual(orderResults(rows, 'name').map(row => row.position), [2, 1, 3]);
  // «Как в файле» — ровно номера файла.
  assert.deepEqual(orderResults(rows, 'original').map(row => row.position), [1, 2, 3]);
});

test('equal values keep file order and the input is not mutated', () => {
  const rows = numberResults([item('А', 0.5), item('Б', 0.5), item('В', 0.5)]);
  const snapshot = rows.map(row => row.position);
  assert.deepEqual(orderResults(rows, 'confidence').map(row => row.position), [1, 2, 3]);
  assert.deepEqual(rows.map(row => row.position), snapshot);
});
