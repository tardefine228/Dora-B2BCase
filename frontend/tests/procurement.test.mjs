import assert from 'node:assert/strict';
import { test } from 'node:test';
import { procurementVerdict } from '../src/procurement.ts';
import { api } from '../src/api.ts';
const result = { status: 'ok', code: '26.20.11', clauseRef: 'Приложение 2, позиция 195', codeRole: 'Включён', catalogReviewRequired: false };

test('ban, restriction and quota describe distinct measures', () => {
  for (const [appendix, title] of [[1, 'Запрет'], [2, 'Ограничение'], [3, 'Минимальная доля']]) {
    const v = procurementVerdict({ ...result, clauseRef: `Приложение ${appendix}, позиция 1` });
    assert.ok(v.title.startsWith(title));
    assert.equal(v.caution, null);
  }
  assert.match(procurementVerdict({ ...result, clauseRef: 'Приложение 3, позиция 37' }).explanation, /223-ФЗ/);
});
test('combined clause references preserve both restriction and quota', () => {
  const v = procurementVerdict({ ...result, clauseRef: 'Приложение 2, позиция 195; Приложение 3, позиция 37' });
  assert.match(v.title, /Ограничение.*Минимальная доля/);
});
test('uncertain matches and missing metadata always show a preliminary warning', () => {
  for (const patch of [{ status: 'review' }, { catalogReviewRequired: true }, { catalogReviewRequired: null }, { codeRole: undefined }]) {
    const v = procurementVerdict({ ...result, ...patch });
    assert.match(v.caution, /Предварительный результат/);
    assert.equal(v.tone, 'neutral');
  }
});
test('no match, exception, absent code and unknown clause never imply permission or ban', () => {
  for (const patch of [{ status: 'not_found' }, { code: null }, { codeRole: 'Исключение' }, { codeRole: 'Не указан' }, { clauseRef: null }, { clauseRef: 'Приложение 4, позиция 1' }, { clauseRef: 'Приложение 12, позиция 1' }, { clauseRef: 'Приложение 2, позиция 1; неизвестная ссылка' }]) {
    const v = procurementVerdict({ ...result, ...patch });
    assert.equal(v.title, 'Условия закупки требуют проверки');
    assert.equal(v.tone, 'neutral');
  }
});
test('API preserves exception and catalogue review evidence for single and batch results', async t => {
  const wire = { query: 'Товар', code: result.code, title: 'Товар', status: 'ok', confidence: .9, clause_ref: result.clauseRef, clause_url: '/source/2-195-1', code_role: 'Исключение', catalog_review_required: true };
  t.mock.method(globalThis, 'fetch', async path => Response.json(path === '/classify' ? wire : { job_id: 'job', status: 'completed', total: 1, completed: 1, results: [wire] }));
  for (const item of [await api.classify('Товар'), ...(await api.getBatch('job')).results]) {
    assert.equal(item.codeRole, 'Исключение');
    assert.equal(item.catalogReviewRequired, true);
    assert.match(procurementVerdict(item).explanation, /исключение из позиции/);
  }
});
