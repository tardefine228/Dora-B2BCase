import assert from 'node:assert/strict';
import { test } from 'node:test';
import { api } from '../src/api.ts';
import { exportResultsCsv } from '../src/files.ts';
import { sanitizeView } from '../src/view.ts';

const original = { code: '29.10.59.140', title: 'Автомобили пожарные', status: 'ok', confidence: 0.8, confidence_raw: 0.8 };
const option = {
  option_id: '1-92-01', code: '29.10.2', title: 'Автомобили легковые', confidence: 0.4,
  okpd2_title: 'Автомобили легковые', okpd2_source: 'https://classifikators.ru/okpd/29.10.2',
  clause_ref: 'Приложение 1, позиция 92', clause_url: '/source/1-92-01',
  code_role: 'Включён', catalog_review_required: false,
};
const pendingRaw = {
  query: 'автомобиль', code: null, title: null, confidence: 0, confidence_raw: 0,
  status: 'needs_clarification', clause_ref: null, clause_url: null, alternatives: [],
  decision_source: 'model', attention_reasons: [{ code: 'ambiguous_name', message: 'Несколько позиций' }],
  item_id: 0, catalog_sha256: 'a'.repeat(64),
  clarification: {
    question: 'Какой именно автомобиль закупается?', catalog_sha256: 'a'.repeat(64), original,
    options: [option, { ...option, option_id: '1-94-01', code: '29.10.4', title: 'Средства автотранспортные грузовые' }],
  },
};

test('API keeps needs_clarification without turning it into review or a code answer', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json(pendingRaw));
  const answer = await api.classify('автомобиль');
  assert.equal(answer.status, 'needs_clarification');
  assert.equal(answer.code, null);
  assert.equal(answer.confidence, null);
  assert.equal(answer.clarification.options.length, 2);
  assert.equal(answer.clarification.options[0].okpd2Title, 'Автомобили легковые');
  assert.equal(answer.clarification.options[0].okpd2Source, option.okpd2_source);
  assert.equal(answer.clarification.original.code, '29.10.59.140');
});

test('single resolution sends only server option id and catalog hash', async t => {
  let request;
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    request = { path, init };
    if (path === '/classify') return Response.json(pendingRaw);
    return Response.json({
      query: 'автомобиль', code: '29.10.2', title: 'Автомобили легковые', confidence: 0.4,
      confidence_raw: 0.4, status: 'ok', clause_ref: option.clause_ref, clause_url: option.clause_url,
      alternatives: [], decision_source: 'specialist', attention_reasons: [],
      resolution: { option_id: option.option_id, resolved_at: '2026-11-11T12:00:00Z', original },
    });
  });
  const pending = await api.classify('автомобиль');
  const resolved = await api.resolve(pending, option.option_id);
  assert.equal(request.path, '/classify/resolve');
  assert.deepEqual(JSON.parse(request.init.body), {
    query: 'автомобиль', option_id: option.option_id, catalog_sha256: 'a'.repeat(64),
  });
  assert.equal(resolved.decisionSource, 'specialist');
  assert.equal(resolved.confidence, null);
  assert.equal(resolved.confidenceRaw, null);
  assert.equal(resolved.resolution.original.code, original.code);
});

test('batch resolution uses stable item id and exports the clarification audit', async t => {
  const paths = [];
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    paths.push(path);
    if (init?.method === 'POST' && path === '/classify') return Response.json(pendingRaw);
    return Response.json({
      ...pendingRaw, status: 'ok', code: option.code, title: option.title, confidence: 0.4,
      clarification: null, decision_source: 'specialist',
      resolution: { option_id: option.option_id, resolved_at: '2026-11-11T12:00:00Z', original },
    });
  });
  const pending = await api.classify('автомобиль');
  const resolved = await api.resolveBatch('job/id', 7, pending, option.option_id);
  assert.equal(paths.at(-1), '/classify/batch/job%2Fid/items/7/resolve');

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
  exportResultsCsv([pending, resolved]);
  const csv = await blob.text();
  assert.ok(csv.includes('clarification_question'));
  assert.ok(csv.includes('Какой именно автомобиль закупается?'));
  assert.ok(csv.includes('29.10.59.140'));
  assert.ok(csv.includes('specialist'));
});

test('stored clarification survives Back navigation, malformed question is dropped', () => {
  const view = { results: [{ ...pendingRaw, name: 'автомобиль', decisionSource: 'model', attentionReasons: [] }] };
  // sanitizeView consumes the frontend camelCase shape, not the raw API shape.
  const frontendPending = {
    name: 'автомобиль', code: null, title: null, status: 'needs_clarification', confidence: null,
    confidenceRaw: 0, clauseRef: null, url: null, alternatives: [], decisionSource: 'model',
    attentionReasons: [], clarification: { catalogSha256: 'a'.repeat(64), options: [{}, {}] },
  };
  assert.equal(sanitizeView({ results: [frontendPending] }).results.length, 1);
  assert.equal(sanitizeView({ results: [{ ...frontendPending, clarification: null }] }).results.length, 0);
  assert.equal(sanitizeView(view).results.length, 0);
});
