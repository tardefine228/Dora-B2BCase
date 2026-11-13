import assert from 'node:assert/strict';
import { test } from 'node:test';
import { addFileHistory, FILE_HISTORY_LIMIT, parseFileHistory, updateFileHistoryStatus } from '../src/history.ts';

const entry = index => ({
  jobId: `job-${index}`, filename: `products-${index}.csv`, total: index + 1,
  createdAt: `2026-11-${String(index + 1).padStart(2, '0')}T12:00:00.000Z`,
  status: 'completed', fileInfo: null,
});

test('file history keeps only the three newest unique reports', () => {
  let history = [];
  for (let index = 0; index < 5; index++) history = addFileHistory(history, entry(index));
  assert.equal(FILE_HISTORY_LIMIT, 3);
  assert.deepEqual(history.map(item => item.jobId), ['job-4', 'job-3', 'job-2']);
  history = addFileHistory(history, { ...entry(3), status: 'running' });
  assert.deepEqual(history.map(item => item.jobId), ['job-3', 'job-4', 'job-2']);
  assert.equal(history[0].status, 'running');
});

test('history status updates without changing order or discarding other reports', () => {
  const history = [entry(2), entry(1), entry(0)];
  const updated = updateFileHistoryStatus(history, 'job-1', 'failed');
  assert.deepEqual(updated.map(item => item.status), ['completed', 'failed', 'completed']);
  assert.deepEqual(updated.map(item => item.jobId), history.map(item => item.jobId));
  assert.notEqual(updated[1], history[1]);
});

test('stored history is sanitized and never needs product rows', () => {
  assert.deepEqual(parseFileHistory(null), []);
  assert.deepEqual(parseFileHistory('{broken'), []);
  assert.deepEqual(parseFileHistory('{"not":"a list"}'), []);
  const parsed = parseFileHistory(JSON.stringify([
    entry(0), { ...entry(1), total: 101 }, { ...entry(2), status: 'unknown' },
    { ...entry(3), fileInfo: { total: 4, warnings: ['ok', 7] } }, 'junk',
  ]));
  assert.deepEqual(parsed.map(item => item.jobId), ['job-0', 'job-3']);
  assert.deepEqual(parsed[1].fileInfo.warnings, ['ok']);
  assert.equal(parsed[1].fileInfo.skipped, 0);
  assert.equal('results' in parsed[1], false);
});
