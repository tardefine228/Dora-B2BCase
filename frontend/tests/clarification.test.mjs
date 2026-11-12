import assert from 'node:assert/strict';
import { test } from 'node:test';
import { clarificationCodeLevel, clarificationOptionExport } from '../src/clarification.ts';

const dumpParent = {
  optionId: '1-77-01', code: '28.92.29', title: 'Автомобили-самосвалы',
  clauseRef: 'Приложение 1, позиция 77', confidence: 0.9, url: null,
  codeRole: 'Включён', catalogReviewRequired: false,
};
const dumpChild = {
  ...dumpParent, optionId: '3-175-01', code: '28.92.29.000',
  clauseRef: 'Приложение 3, позиция 175',
};
const microscope = {
  ...dumpParent, optionId: '2-379-01', code: '26.60.12.119',
  title: 'Микроскопы медицинские', clauseRef: 'Приложение 2, позиция 379',
};

test('code hierarchy is derived only from explicit parent/child options', () => {
  const options = [dumpParent, dumpChild, microscope];
  assert.equal(clarificationCodeLevel(dumpParent, options), 'group');
  assert.equal(clarificationCodeLevel(dumpChild, options), 'detail');
  assert.equal(clarificationCodeLevel(microscope, options), null);
  assert.equal(clarificationCodeLevel(dumpParent, [dumpParent]), null);
});

test('export distinguishes identical titles by clause and code level', () => {
  const options = [dumpParent, dumpChild];
  assert.equal(
    clarificationOptionExport(dumpParent, options),
    '1-77-01: 28.92.29 — Автомобили-самосвалы '
      + '[Приложение 1, позиция 77; Укрупнённый код]',
  );
  assert.equal(
    clarificationOptionExport(dumpChild, options, false),
    '28.92.29.000 — Автомобили-самосвалы '
      + '[Приложение 3, позиция 175; Детальный код]',
  );
  assert.ok(clarificationOptionExport(microscope, [microscope]).includes('Приложение 2'));
  assert.ok(!clarificationOptionExport(microscope, [microscope]).includes('Детальный код'));
});
