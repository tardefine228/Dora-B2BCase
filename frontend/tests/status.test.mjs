import assert from 'node:assert/strict';
import { test } from 'node:test';
import { STATUS_HELP, STATUS_LABEL, codeLabel, duplicateAppendixHint, notFoundHint } from '../src/status.ts';

const alternative = (code, title, confidence = 0.5) => ({
  code, title, confidence, clauseRef: null, url: null,
});

test('status labels separate "нашёл, но не уверен" from "нет в перечне"', () => {
  // Эксп. 30: в полосе review 26 из 29 строк имели верный код, поэтому подпись
  // не должна читаться как «ничего не найдено».
  assert.equal(STATUS_LABEL.review, 'Предварительный результат — проверьте позицию');
  assert.equal(STATUS_LABEL.not_found, 'Совпадение не найдено');
  assert.notEqual(STATUS_LABEL.review, STATUS_LABEL.not_found);
  assert.ok(STATUS_HELP.review.includes('Код-кандидат'));
  assert.ok(!STATUS_HELP.review.toLowerCase().includes('не найдено'));
});

test('notFoundHint tells apart "нет похожих" and "есть похожие"', () => {
  const empty = notFoundHint({ alternatives: [] });
  assert.ok(empty.includes('Похожих позиций нет'));
  const withHints = notFoundHint({ alternatives: [alternative('26.20.2', 'Устройства запоминающие')] });
  assert.ok(withHints.includes('Похожие позиции'));
  assert.ok(withHints.includes('подсказка'));
});

test('the same position in another appendix is called out instead of being hidden', () => {
  // 81 семейство позиций повторяется в приложениях 2 и 3 под разными кодами.
  const result = {
    code: '31.01.1',
    title: 'Мебель для офисов и предприятий торговли',
    alternatives: [alternative('31.01.12', 'Мебель для офисов и предприятий торговли')],
  };
  const hint = duplicateAppendixHint(result);
  assert.ok(hint && hint.includes('31.01.12'));
  assert.ok(hint.includes('приложением'));

  // Разные позиции — не двойник: подсказки нет.
  assert.equal(duplicateAppendixHint({
    code: '26.20.2', title: 'Устройства запоминающие',
    alternatives: [alternative('26.20.15', 'Машины вычислительные')],
  }), null);
  // Молчание без кода тоже не даёт подсказки о приложении.
  assert.equal(duplicateAppendixHint({ code: null, title: null, alternatives: [] }), null);
});

test('code label marks a review answer as a candidate', () => {
  assert.equal(codeLabel('review'), 'Код-кандидат (проверьте)');
  assert.ok(codeLabel('ok').includes('ОКПД2'));
});

test('a verified code explains itself instead of showing a fake percentage', async () => {
  const { verifierNote } = await import('../src/status.ts');
  const note = verifierNote({ verifier: 'llm-top3', verifierNote: 'код подтверждён по top-3' });
  assert.ok(note.includes('проверяющей моделью'));
  assert.ok(note.includes('проверки специалистом'));
  assert.equal(verifierNote({ verifier: null }), null);
});
