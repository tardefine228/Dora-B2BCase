import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');

test('old product history is gone; the new file history stores metadata only', () => {
  // Старая история хранила названия товаров. Новый блок по явному запросу
  // пользователя содержит только имя файла, opaque job id и статус отчёта.
  assert.ok(!app.includes('История проверок на этом устройстве'));
  assert.ok(!app.includes('history-optin'));
  assert.ok(!app.includes('Сохранять новые проверки в истории'));
  assert.ok(!app.includes('HISTORY_ENABLED_KEY'));
  assert.ok(!app.includes('saveHistory'));
  assert.ok(app.includes('История файлов'));
  assert.ok(app.includes('без названий товаров'));
  assert.ok(styles.includes('.file-history'));
});

test('records left in the browser by the removed feature are cleaned up', () => {
  // У тех, кто успел включить галочку, названия товаров лежали в localStorage.
  // Эти старые ключи продолжаем стирать при открытии страницы.
  assert.ok(app.includes("'dora:local-history:v1'"));
  assert.ok(app.includes("'dora:history-enabled:v1'"));
  assert.ok(app.includes('storage.removeItem(key)'));
});
