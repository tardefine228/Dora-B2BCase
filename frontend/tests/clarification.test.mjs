import assert from 'node:assert/strict';
import { test } from 'node:test';
import { clarificationCodeLevel, clarificationOptionExport, groupClarificationOptions, groupClarificationTitles, clarificationGuide } from '../src/clarification.ts';

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

test('identical code and title become one choice with all source positions', () => {
  const second = { ...dumpParent, optionId: '2-200-01', clauseRef: 'Приложение 2, позиция 200' };
  const groups = groupClarificationOptions([dumpParent, second]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].code, '28.92.29');
  assert.deepEqual(groups[0].options.map(option => option.optionId), ['1-77-01', '2-200-01']);
  assert.deepEqual(groups[0].clauseRefs, ['Приложение 1, позиция 77', 'Приложение 2, позиция 200']);
});

test('equal codes with different legal descriptions are never merged', () => {
  const other = { ...dumpParent, optionId: '2-999-01', title: 'Другой товар' };
  const groups = groupClarificationOptions([dumpParent, other]);
  assert.equal(groups.length, 2);
});

test('one official title becomes one card with nested code choices', () => {
  const butter = { ...dumpParent, code: '10.51.30.100', title: 'Масло сливочное' };
  const otherButter = { ...butter, optionId: '2-451-01', code: '10.51.30.110', clauseRef: 'Приложение 2, позиция 451' };
  const groups = groupClarificationTitles([butter, otherButter, dumpParent]);
  assert.equal(groups.length, 2);
  const butterGroup = groups.find(group => group.title === 'Масло сливочное');
  assert.equal(butterGroup.codeGroups.length, 2);
  assert.deepEqual(butterGroup.codeGroups.map(group => group.code), ['10.51.30.100', '10.51.30.110']);
  assert.equal(butterGroup.clauseRefs.length, 2);
});

test('already combined clause references are deduplicated inside one group', () => {
  const combined = { ...dumpParent, clauseRef: 'Приложение 1, позиция 77; Приложение 3, позиция 175' };
  const groups = groupClarificationOptions([combined, dumpParent]);
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].clauseRefs, ['Приложение 1, позиция 77', 'Приложение 3, позиция 175']);
});

test('guidance describes known categories without inventing unknown distinctions', () => {
  assert.equal(clarificationGuide({ code: '21.20.24.120', okpd2Title: 'Шовные материалы' }).title, 'Шовные материалы');
  assert.equal(clarificationGuide({ code: '21.20.24' }), null);
  assert.equal(clarificationGuide({ code: 'toString' }), null);
});

test('butter category and subcategory are shown as hierarchy, not fat percentage', () => {
  const parent = { ...dumpParent, code: '10.51.30.100' };
  const child = { ...dumpChild, code: '10.51.30.110' };
  assert.equal(clarificationCodeLevel(parent, [parent, child]), 'group');
  assert.equal(clarificationCodeLevel(child, [parent, child]), 'detail');
  assert.equal(clarificationCodeLevel(parent, [parent, { ...child, code: '10.51.30.200' }]), null);
});
