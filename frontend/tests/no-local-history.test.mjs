import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
const styles = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');

test('the local history block is gone from the interface', () => {
  // Продуктовое решение: локальная история проверок не нужна. Она хранила
  // названия товаров в браузере, ничего не давала для работы классификатора и
  // занимала место внизу страницы, вызывая вопрос «а это про что?».
  assert.ok(!app.includes('История проверок на этом устройстве'), 'блок истории вернулся в App.tsx');
  assert.ok(!app.includes('history-optin'), 'галочка истории осталась в разметке');
  assert.ok(!app.includes('Сохранять новые проверки в истории'), 'текст про историю остался');
  assert.ok(!app.includes('HISTORY_ENABLED_KEY'), 'состояние истории осталось в коде');
  assert.ok(!app.includes('saveHistory'), 'запись в историю осталась');
  assert.ok(!styles.includes('.history'), 'стили истории остались в styles.css');
});

test('records left in the browser by the removed feature are cleaned up', () => {
  // У тех, кто успел включить галочку, названия товаров лежат в localStorage.
  // Без UI они остались бы там навсегда, поэтому ключи стираются при открытии.
  assert.ok(app.includes("'dora:local-history:v1'"));
  assert.ok(app.includes("'dora:history-enabled:v1'"));
  assert.ok(app.includes('storage.removeItem(key)'));
});
