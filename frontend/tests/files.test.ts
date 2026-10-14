import test from 'node:test';
import assert from 'node:assert/strict';
import { makeTable, namesFrom, switchHeader, switchSheet } from '../src/files.ts';

test('recognizes full Russian header and does not classify row numbers', () => {
  const t = makeTable([['№', 'Наименование товара', 'Цена'], ['1', 'Ноутбук', '50000']]);
  assert.equal(t.column, 1);
  assert.deepEqual(namesFrom(t, t.column), ['Ноутбук']);
});
test('preserves first product in headerless and one-row files', () => {
  const t = makeTable([['Ноутбук'], ['Мышь']]);
  assert.deepEqual(namesFrom(t, 0), ['Ноутбук', 'Мышь']);
  assert.deepEqual(namesFrom(makeTable([['Ноутбук']]), 0), ['Ноутбук']);
});
test('selects text instead of numeric SKU in a headerless file', () => {
  const t = makeTable([['120011', 'Ноутбук', '50000'], ['120012', 'Клавиатура', '3000']]);
  assert.equal(t.column, 1);
  assert.deepEqual(namesFrom(t, t.column), ['Ноутбук', 'Клавиатура']);
});
test('allows explicit header override and preserves duplicates', () => {
  const t = switchHeader(makeTable([['Артикул'], ['Ноутбук'], ['Ноутбук']]), true);
  assert.deepEqual(namesFrom(t, 0), ['Ноутбук', 'Ноутбук']);
});
test('switches sheets and rejects more than 100 items without truncation', () => {
  const sheets = [{name:'A',raw:[['Ноутбук']]},{name:'B',raw:[['Мышь']]}];
  assert.deepEqual(namesFrom(switchSheet(makeTable(sheets[0].raw,sheets),1),0), ['Мышь']);
  assert.throws(() => namesFrom(makeTable(Array.from({length:101}, () => ['Мышь'])),0));
});
