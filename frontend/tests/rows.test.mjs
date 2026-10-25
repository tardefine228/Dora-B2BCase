import assert from 'node:assert/strict';
import { test } from 'node:test';
import { orderResults } from '../src/rows.ts';

const item = (name, confidence) => ({
  name, confidence, code: null, title: null, status: 'review',
  confidenceRaw: confidence, clauseRef: null, url: null, alternatives: [],
});
// Порядок файла: «Ноутбук», «Бумага», «Стол».
const file = [item('Ноутбук', 0.4), item('Бумага', 0.9), item('Стол', 0.6)];

test('«Как в файле» keeps the server order', () => {
  assert.deepEqual(orderResults(file, 'original').map(r => r.name), ['Ноутбук', 'Бумага', 'Стол']);
});

test('sorting reorders rows and never mutates the input', () => {
  const snapshot = file.map(r => r.name);
  assert.deepEqual(orderResults(file, 'confidence').map(r => r.name), ['Бумага', 'Стол', 'Ноутбук']);
  assert.deepEqual(orderResults(file, 'name').map(r => r.name), ['Бумага', 'Ноутбук', 'Стол']);
  assert.deepEqual(file.map(r => r.name), snapshot);
});

test('equal values keep the file order, so the result is deterministic', () => {
  const same = [item('А', 0.5), item('Б', 0.5), item('В', 0.5)];
  assert.deepEqual(orderResults(same, 'confidence').map(r => r.name), ['А', 'Б', 'В']);
});

test('the number column is a plain counter 1…N over the shown order', () => {
  // Колонку «№» рисует таблица: index + 1 в текущем порядке, без файловых номеров.
  const numbers = sort => orderResults(file, sort).map((_result, index) => index + 1);
  assert.deepEqual(numbers('original'), [1, 2, 3]);
  assert.deepEqual(numbers('name'), [1, 2, 3]);
  assert.deepEqual(numbers('confidence'), [1, 2, 3]);
});
