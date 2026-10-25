import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  VIEW_KEY, isRunningJob, navigationType, parseJobMarker, readStoredView, sanitizeView, shouldRestoreView,
} from '../src/view.ts';

function memoryStorage(initial) {
  const data = new Map(Object.entries(initial ?? {}));
  return {
    data,
    getItem: key => data.get(key) ?? null,
    removeItem: key => { data.delete(key); },
  };
}

const savedResults = [
  { name: 'Ноутбук Lenovo', code: '26.20.11', title: null, confidence: 0.9, confidenceRaw: 0.9, status: 'ok', clauseRef: null, url: null, alternatives: [] },
];

test('a fresh site launch does not show the previously analysed file', () => {
  // Баг: sessionStorage переживал перезапуск браузера, и страница открывалась с
  // уже разобранным файлом, который в этой сессии не запускали.
  for (const type of ['navigate', 'reload', 'prerender']) {
    const session = memoryStorage({
      [VIEW_KEY]: JSON.stringify({ filename: 'products.csv', results: savedResults, progress: { processed: 3, total: 3, state: 'done' } }),
    });
    assert.equal(readStoredView(session, type), null, type);
    // И запись стёрта, иначе она всплывёт при следующем возврате по «Назад».
    assert.equal(session.data.has(VIEW_KEY), false, type);
  }
});

test('coming back with the browser Back button still restores the view', () => {
  const session = memoryStorage({
    [VIEW_KEY]: JSON.stringify({ filename: 'products.csv', results: savedResults, name: 'Ноутбук' }),
  });
  const view = readStoredView(session, 'back_forward');
  assert.equal(view.filename, 'products.csv');
  assert.equal(view.results.length, 1);
  assert.equal(view.name, 'Ноутбук');
  assert.equal(session.data.has(VIEW_KEY), true);
});

test('only back_forward counts as a return to the page', () => {
  assert.equal(shouldRestoreView('back_forward'), true);
  assert.equal(shouldRestoreView('navigate'), false);
  assert.equal(shouldRestoreView('reload'), false);
});

test('navigation type falls back to a clean start when the browser says nothing', () => {
  assert.equal(navigationType([]), 'navigate');
  assert.equal(navigationType(undefined), 'navigate');
  assert.equal(navigationType([{ type: 'back_forward' }]), 'back_forward');
});

test('a corrupted or foreign record is dropped instead of rendered', () => {
  const broken = memoryStorage({ [VIEW_KEY]: '{not json' });
  assert.equal(readStoredView(broken, 'back_forward'), null);
  assert.equal(broken.data.has(VIEW_KEY), false);

  const partial = sanitizeView({ results: 'nope', progress: { state: 'running' }, unknown: 1 });
  assert.deepEqual(partial, {});
  assert.equal(sanitizeView(null).results, undefined);
  const mixed = sanitizeView({ results: [savedResults[0], 'junk', null], filename: 42 });
  assert.equal(mixed.results.length, 1);
  assert.equal(mixed.filename, undefined);
});

test('a partially written record never reaches the renderer broken', () => {
  // UI вызывает fileInfo.warnings.map и result.alternatives.length: запись без
  // этих полей давала белый экран вместо страницы.
  const view = sanitizeView({
    fileInfo: { total: 12, source_column: 'Наименование', warnings: ['часть строк пропущена', 7] },
    progress: { processed: 4, total: 12, state: 'running' },
    results: [{ name: 'Стол', status: 'ok' }],
    result: { name: 'Стул', status: 'review' },
  });
  assert.deepEqual(view.fileInfo.warnings, ['часть строк пропущена']);
  assert.equal(view.fileInfo.skipped, 0);
  assert.equal(view.fileInfo.truncated, false);
  assert.deepEqual(view.results[0].alternatives, []);
  assert.deepEqual(view.result.alternatives, []);
  assert.equal(view.progress.state, 'running');
  // Неизвестный статус и битый прогресс — не восстанавливаем вовсе.
  assert.equal(sanitizeView({ results: [{ name: 'Стол', status: 'weird' }] }).results.length, 0);
  assert.equal(sanitizeView({ progress: { processed: 1, total: 0, state: 'done' } }).progress, undefined);
  assert.equal(sanitizeView({ progress: { processed: 1, total: 3, state: 'paused' } }).progress, undefined);
  assert.equal(sanitizeView({ fileInfo: { warnings: [] } }).fileInfo, undefined);
});

test('only a running job is resumed on startup', () => {
  assert.equal(isRunningJob('queued'), true);
  assert.equal(isRunningJob('running'), true);
  // Завершённое задание — это результат чужой сессии: его не показываем.
  assert.equal(isRunningJob('completed'), false);
  assert.equal(isRunningJob('failed'), false);
});

test('the stored job marker is validated before use', () => {
  assert.deepEqual(parseJobMarker(JSON.stringify({ id: 'job-1', total: 3, filename: 'a.csv' })),
    { id: 'job-1', total: 3, filename: 'a.csv' });
  assert.equal(parseJobMarker(null), null);
  assert.equal(parseJobMarker('{oops'), null);
  assert.equal(parseJobMarker(JSON.stringify({ id: '', total: 3 })), null);
  assert.equal(parseJobMarker(JSON.stringify({ id: 'job-1', total: 0 })), null);
  assert.equal(parseJobMarker(JSON.stringify({ id: 'job-1', total: 101 })), null);
  assert.equal(parseJobMarker(JSON.stringify({ id: 'job-1', total: 1.5 })), null);
});
