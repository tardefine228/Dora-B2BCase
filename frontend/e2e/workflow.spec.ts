import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
const result = (query: string, code: string) => ({ query, code, title: 'Категория товара', status: 'ok', confidence: .95, confidence_raw: .8, clause_ref: 'Пункт 1', clause_url: '/source/test-1', alternatives: [] });

test('single query, keyboard submission, feedback and accessibility', async ({ page }) => {
  await page.route('**/classify', r => r.fulfill({ json: result('Ноутбук', '26.20.11') }));
  let feedback: any;
  await page.route('**/feedback', r => { feedback = r.request().postDataJSON(); return r.fulfill({ json: { saved: true } }); });
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Ноутбук');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.getByText('26.20.11', { exact: true })).toBeVisible();
  await expect(page.getByText('Технические сведения')).toHaveCount(0);
  await page.getByText('Сообщить об ошибке (для экспертов)', { exact: true }).click();
  await page.getByRole('button', { name: 'Код верный' }).click();
  await expect(page.getByRole('status')).toContainText('сохранена');
  expect(feedback.query).toBe('Ноутбук');
  const audit = await new AxeBuilder({ page }).analyze();
  expect(audit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
});

test('file results survive Back and feedback stays with its row after sorting', async ({ page, context }) => {
  const rows = [result('Стол', '31.01'), result('Бумага', '17.12')];
  let uploads = 0;
  await page.route('**/classify/file', r => { uploads++; return r.fulfill({ json: { job_id: 'qa', total: 2, total_found: 2, filename: 'goods.csv', format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } }); });
  await page.route('**/classify/batch/qa/events', r => r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' }));
  await page.route('**/classify/batch/qa', r => r.fulfill({ json: { job_id: 'qa', status: 'completed', total: 2, completed: 2, results: rows } }));
  await page.route('**/source/test-1?*', r => r.fulfill({ contentType: 'text/html', body: '<h1>Пункт постановления</h1>' }));
  await page.goto('/ui/');
  await page.locator('input[type=file]').setInputFiles({ name: 'goods.csv', mimeType: 'text/csv', buffer: Buffer.from('name\nСтол\nБумага') });
  await expect(page.locator('tbody tr')).toHaveCount(2);
  const tableRow = page.locator('tbody tr').filter({ hasText: 'Стол' });
  await tableRow.getByText('Проверить результат', { exact: true }).click();
  await tableRow.getByText('Сообщить об ошибке (для экспертов)', { exact: true }).click();
  await tableRow.getByRole('button', { name: 'Код неверный' }).click();
  await tableRow.getByLabel('Правильный код ОКПД2').fill('31.02');
  await page.getByLabel('Сортировка').selectOption('name');
  await expect(tableRow.getByLabel('Правильный код ОКПД2')).toHaveValue('31.02');
  await tableRow.getByRole('link').first().click();
  await expect(page).toHaveURL(/source/);
  expect(context.pages()).toHaveLength(1);
  await page.goBack();
  await expect(page.locator('tbody tr')).toHaveCount(2);
  expect(uploads).toBe(1);
});

test('mobile page has no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/ui/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});


test('not found displays confidence without suggesting certainty of absence', async ({ page }) => {
  await page.route('**/classify', r => r.fulfill({ json: {
    query: 'Неизвестный товар', code: null, title: null, status: 'not_found',
    confidence: 0, confidence_raw: 0, clause_ref: null, clause_url: null, alternatives: [],
  } }));
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Неизвестный товар');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.locator('.confidence strong')).toHaveText('0%');
  await expect(page.locator('.confidence')).toContainText('Уверенность в подборе кода');
  await expect(page.locator('.status-help')).toContainText('не вероятность отсутствия');
});
