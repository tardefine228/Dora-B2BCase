import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { test } from 'node:test';
import { api, followBatch } from '../src/api.ts';
import { storage } from '../src/storage.ts';
import { exportResultsCsv } from '../src/files.ts';

const batch = (status = 'completed') => ({
  job_id: 'job', status, completed: status === 'completed' ? 1 : 0,
  total: 1, results: [], error: null,
});

function environment(t, fetcher, sse = true) {
  const timers = new Map();
  let nextTimer = 0;
  class Source extends EventTarget {
    static instances = [];
    closed = false;
    constructor() { super(); Source.instances.push(this); }
    close() { this.closed = true; }
  }
  t.mock.method(globalThis, 'fetch', fetcher);
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    ...(sse ? { EventSource: Source } : {}),
    setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
  } });
  const previousSource = Object.getOwnPropertyDescriptor(globalThis, 'EventSource');
  Object.defineProperty(globalThis, 'EventSource', { configurable: true, value: Source });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'window', previous); else delete globalThis.window;
    if (previousSource) Object.defineProperty(globalThis, 'EventSource', previousSource); else delete globalThis.EventSource;
  });
  return { timers, Source };
}

test('blocked storage and full quota do not throw', t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get() { throw new Error('blocked'); } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous); else delete globalThis.localStorage;
  });
  assert.equal(storage.getItem('history'), null);
  assert.equal(storage.setItem('job', '1'), false);
  assert.equal(storage.removeItem('job'), false);
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: () => 'saved', setItem() { throw new Error('quota'); }, removeItem() {},
  } });
  assert.equal(storage.getItem('history'), 'saved');
  assert.equal(storage.setItem('job', '1'), false);
  assert.equal(storage.removeItem('job'), true);
});

test('SSE cancellation closes connection and removes timeout and abort listener', async t => {
  const { timers, Source } = environment(t, async () => { throw new Error('unexpected fetch'); });
  const controller = new AbortController();
  const pending = followBatch('job', () => assert.fail('unexpected progress'), controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(Source.instances[0].closed, true);
  assert.equal(timers.size, 0);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});

test('already aborted tracking does not open a connection', async t => {
  const { Source } = environment(t, async () => { throw new Error('unexpected fetch'); });
  await assert.rejects(followBatch('job', () => {}, AbortSignal.abort()), { name: 'AbortError' });
  assert.equal(Source.instances.length, 0);
});

test('cancelling final SSE fetch does not deliver stale results', async t => {
  let resolveFetch;
  const { Source, timers } = environment(t, () => new Promise(resolve => { resolveFetch = resolve; }));
  const controller = new AbortController();
  const pending = followBatch('job', () => assert.fail('stale progress'), controller.signal);
  Source.instances[0].dispatchEvent(new Event('done'));
  await Promise.resolve();
  controller.abort();
  resolveFetch(Response.json(batch()));
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(timers.size, 0);
});

test('SSE failure falls back to polling and returns failed jobs', async t => {
  const { Source, timers } = environment(t, async () => Response.json(batch('failed')));
  const controller = new AbortController();
  const updates = [];
  const pending = followBatch('job', job => updates.push(job), controller.signal);
  Source.instances[0].dispatchEvent(new Event('error'));
  assert.equal((await pending).status, 'failed');
  assert.equal(updates.length, 1);
  assert.equal(timers.size, 0);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});

test('polling removes wait listeners after each tick', async t => {
  let calls = 0;
  const { timers } = environment(t, async () => Response.json(batch(++calls < 3 ? 'running' : 'completed')), false);
  const controller = new AbortController();
  const pending = followBatch('job', () => {}, controller.signal);
  for (let i = 0; i < 2; i++) {
    while (!timers.size) await new Promise(resolve => setImmediate(resolve));
    assert.equal(getEventListeners(controller.signal, 'abort').length, 1);
    [...timers.values()][0]();
    assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  }
  assert.equal((await pending).status, 'completed');
  assert.equal(timers.size, 0);
});

test('API errors retain validation messages', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: [{ msg: 'Name required' }] }, { status: 422 }));
  await assert.rejects(api.classify(''), /Name required/);
});

// Перехватывает файл, который экспорт отдаёт браузеру: возвращает геттер блоба.
function captureDownload(t) {
  let blob;
  t.mock.method(URL, 'createObjectURL', value => { blob = value; return 'blob:test'; });
  t.mock.method(globalThis, 'setTimeout', () => 0);
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    body: { appendChild() {} }, createElement() { return { click() {}, remove() {} }; },
  } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'document', previous); else delete globalThis.document;
  });
  return () => blob;
}

test('CSV neutralizes formulas and preserves delimiter and quote escaping', async t => {
  const capture = captureDownload(t);
  const names = ['=1+1', '+1', '-1', '@SUM(A1)', '  =1', '\tformula', 'Обычный; "товар"'];
  exportResultsCsv(names.map(name => ({
    name, code: '27.51.24.110', title: null, status: 'ok', confidence: 0.9,
    confidenceRaw: -0.1, clauseRef: null, url: null,
  })));
  const csv = await capture().text();
  for (const name of names.slice(0, -1)) assert.ok(csv.includes(`'${name};`));
  assert.ok(csv.includes('"Обычный; ""товар""";27.51.24.110;'));
  assert.ok(csv.includes(';0.9;-0.1;'));
});

test('CSV preserves row numbers and includes procurement explanation', async t => {
  const capture = captureDownload(t);
  exportResultsCsv([
    { name: 'Второй', code: null, title: null, status: 'not_found', confidence: 0, confidenceRaw: 0, clauseRef: null, url: null },
    { name: 'Первый', code: '26.20.11', title: null, status: 'ok', confidence: 0.9, confidenceRaw: 0.8, clauseRef: null, url: null },
  ]);
  const lines = (await capture().text()).trim().split('\r\n');
  assert.equal(lines[0].split(';')[12], 'row');
  assert.equal(lines[1].split(';')[12], '1');
  assert.equal(lines[2].split(';')[12], '2');
  assert.ok(lines[0].endsWith(';procurement_regime;procurement_explanation'));
  assert.ok(lines[1].includes('Это не подтверждает'));
});


test('not-found retains model confidence in UI and CSV without claiming a code', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({
    query: 'Неизвестный товар', code: null, title: null, status: 'not_found',
    confidence: 0, confidence_raw: 0, clause_ref: null, clause_url: null, alternatives: [],
  }));
  const answer = await api.classify('Неизвестный товар');
  assert.equal(answer.confidence, 0);
  assert.equal(answer.code, null);
  assert.equal(answer.url, null);
  const capture = captureDownload(t);
  exportResultsCsv([answer]);
  assert.ok((await capture().text()).includes(';not_found;0;'));
});
