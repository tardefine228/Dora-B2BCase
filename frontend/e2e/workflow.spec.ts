import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
const result = (query: string, code: string) => ({ query, code, title: 'Категория товара', status: 'ok', confidence: .95, confidence_raw: .8, clause_ref: 'Пункт 1', clause_url: '/source/test-1', alternatives: [] });
const original = { code: '29.10.59.140', title: 'Автомобили пожарные', status: 'ok', confidence: .8, confidence_raw: .8, alternatives: [] };
const clarificationOptions = [
  { option_id: '1-92-01', code: '29.10.2', title: 'Автомобили легковые', confidence: .4, clause_ref: 'Приложение 1, позиция 92', clause_url: '/source/1-92-01', code_role: 'Включён', catalog_review_required: false },
  { option_id: '1-94-01', code: '29.10.4', title: 'Средства автотранспортные грузовые', confidence: .3, clause_ref: 'Приложение 1, позиция 94', clause_url: '/source/1-94-01', code_role: 'Включён', catalog_review_required: false },
];
const pending = (itemId: number | null = null) => ({
  query: 'автомобиль', code: null, title: null, status: 'needs_clarification', confidence: 0, confidence_raw: 0,
  clause_ref: null, clause_url: null, alternatives: [], item_id: itemId, decision_source: 'model', attention_reasons: [],
  clarification: { question: 'Какой именно автомобиль закупается?', options: clarificationOptions, catalog_sha256: 'a'.repeat(64), original },
});

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
  const rows = [result('Стол', '31.01'), result('Бумага', '17.12')].map(row => ({ ...row, clause_ref: 'Приложение 2, позиция 1', code_role: 'Включён', catalog_review_required: false }));
  let uploads = 0;
  await page.route('**/classify/file', r => { uploads++; return r.fulfill({ json: { job_id: 'qa', total: 2, total_found: 2, filename: 'goods.csv', format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } }); });
  await page.route('**/classify/batch/qa/events', r => r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' }));
  await page.route('**/classify/batch/qa', r => r.fulfill({ json: { job_id: 'qa', status: 'completed', total: 2, completed: 2, results: rows } }));
  await page.route('**/source/test-1?*', r => r.fulfill({ contentType: 'text/html', body: '<h1>Пункт постановления</h1>' }));
  await page.goto('/ui/');
  await page.locator('input[type=file]').setInputFiles({ name: 'goods.csv', mimeType: 'text/csv', buffer: Buffer.from('name\nСтол\nБумага') });
  await expect(page.locator('tbody tr')).toHaveCount(2);
  const tableRow = page.locator('tbody tr').filter({ hasText: 'Стол' });
  await expect(tableRow.locator('.procurement-brief')).toContainText('Ограничение закупки');
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


test('procurement verdict shows both appendices and the supplied flag on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/classify', r => r.fulfill({ json: {
    ...result('Ноутбук', '26.20.11'), clause_ref: 'Приложение 2, позиция 195; Приложение 3, позиция 37',
    code_role: 'Включён', catalog_review_required: false,
  } }));
  await page.goto('/ui/');
  await expect(page.getByAltText('Флаг России')).toBeVisible();
  const favicon = await page.locator('link[rel="icon"]').getAttribute('href');
  expect(favicon).toBe('/ui/russia-flag.svg');
  expect((await page.request.get(favicon!)).status()).toBe(200);
  await page.getByLabel('Название товара', { exact: true }).fill('Ноутбук');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  const verdict = page.locator('.procurement');
  await expect(verdict).toContainText('Ограничение закупки');
  await expect(verdict).toContainText('Минимальная доля');
  await expect(verdict).toContainText('223-ФЗ');
  await expect(page.locator('footer')).toContainText('Для закупок по 44-ФЗ и 223-ФЗ');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('ambiguous single query has no code and resolves in one click', async ({ page }) => {
  let resolution: any;
  await page.route('**/classify', r => r.fulfill({ json: pending() }));
  await page.route('**/classify/resolve', r => {
    resolution = r.request().postDataJSON();
    return r.fulfill({ json: {
      ...result('автомобиль', '29.10.2'), title: 'Автомобили легковые', confidence: .4,
      code_role: 'Включён', catalog_review_required: false, decision_source: 'specialist',
      resolution: { option_id: '1-92-01', resolved_at: '2026-11-11T12:00:00Z', original },
    } });
  });
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('автомобиль');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.getByText('Код не назначен до уточнения')).toBeVisible();
  await expect(page.locator('.code')).toHaveCount(0);
  await expect(page.getByText('Какой именно автомобиль закупается?')).toBeVisible();
  const pendingAudit = await new AxeBuilder({ page }).analyze();
  expect(pendingAudit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
  await page.getByRole('button', { name: /29\.10\.2.*Автомобили легковые/ }).click();
  await expect(page.locator('.code')).toHaveText('29.10.2');
  await expect(page.locator('.resolution-note')).toContainText('Выбрано специалистом');
  await expect(page.locator('.confidence')).toContainText('Ручной выбор');
  await expect(page.locator('.confidence progress')).toHaveCount(0);
  expect(resolution).toEqual({ query: 'автомобиль', option_id: '1-92-01', catalog_sha256: 'a'.repeat(64) });
  const audit = await new AxeBuilder({ page }).analyze();
  expect(audit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
});

test('batch clarification resolves the stable server item after sorting', async ({ page }) => {
  const rows = [pending(0), { ...result('Бумага', '17.12'), item_id: 1 }];
  let resolvePath = '';
  let resolved = false;
  const batchReport = () => ({
    summary: { total: 2, accepted: resolved ? 2 : 1, auto_accepted: 1, specialist_resolved: resolved ? 1 : 0, needs_review: resolved ? 0 : 1, not_found: 0, reason_counts: resolved ? {} : { ambiguous_name: 1 } },
    items: [
      { item_id: 0, bucket: resolved ? 'accepted' : 'needs_review', risk_rank: resolved ? 0 : 90, reasons: resolved ? [] : [{ code: 'ambiguous_name', message: 'Нужен выбор специалиста.' }] },
      { item_id: 1, bucket: 'accepted', risk_rank: 0, reasons: [] },
    ],
  });
  await page.route('**/classify/file', r => r.fulfill({ json: { job_id: 'ambiguity', total: 2, total_found: 2, filename: 'goods.csv', format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } }));
  await page.route('**/classify/batch/ambiguity/events', r => r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' }));
  await page.route('**/classify/batch/ambiguity', r => r.fulfill({ json: { job_id: 'ambiguity', status: 'completed', total: 2, completed: 2, results: rows, report: batchReport() } }));
  await page.route('**/classify/batch/ambiguity/items/0/resolve', r => {
    resolvePath = r.request().url();
    resolved = true;
    return r.fulfill({ json: { ...result('автомобиль', '29.10.2'), item_id: 0, decision_source: 'specialist', resolution: { option_id: '1-92-01', resolved_at: '2026-11-11T12:00:00Z', original } } });
  });
  await page.goto('/ui/');
  await page.locator('input[type=file]').setInputFiles({ name: 'goods.csv', mimeType: 'text/csv', buffer: Buffer.from('name\nавтомобиль\nБумага') });
  await expect(page.locator('tbody tr')).toHaveCount(2);
  await page.getByLabel('Сортировка').selectOption('name');
  const row = page.locator('tbody tr').filter({ hasText: 'автомобиль' });
  await row.getByText('Уточнить товар', { exact: true }).click();
  await row.getByRole('button', { name: /29\.10\.2.*Автомобили легковые/ }).click();
  await expect(row.locator('td').nth(2)).toContainText('29.10.2');
  await expect(page.locator('.summary-accepted strong')).toHaveText('2');
  await expect(page.locator('.summary-needs_review strong')).toHaveText('0');
  expect(resolvePath).toContain('/classify/batch/ambiguity/items/0/resolve');
});

test('procurement summary filters risks and completed report survives reload', async ({ page }) => {
  const rows = [
    pending(0),
    { ...result('Бумага', '17.12'), item_id: 1 },
    { ...result('Неизвестный товар', ''), code: null, title: null, status: 'not_found', confidence: 0, item_id: 2 },
  ];
  const report = {
    summary: { total: 3, accepted: 1, auto_accepted: 1, specialist_resolved: 0, needs_review: 1, not_found: 1, reason_counts: { ambiguous_name: 1, not_found: 1 } },
    items: [
      { item_id: 0, bucket: 'needs_review', risk_rank: 90, reasons: [{ code: 'ambiguous_name', message: 'Нужен выбор специалиста.' }] },
      { item_id: 1, bucket: 'accepted', risk_rank: 0, reasons: [] },
      { item_id: 2, bucket: 'not_found', risk_rank: 60, reasons: [{ code: 'not_found', message: 'Совпадение не найдено.' }] },
    ],
  };
  let uploads = 0;
  await page.route('**/classify/file', r => { uploads++; return r.fulfill({ json: { job_id: 'report', total: 3, total_found: 3, filename: 'goods.csv', format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } }); });
  await page.route('**/classify/batch/report/events', r => r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' }));
  await page.route('**/classify/batch/report', r => r.fulfill({ json: { job_id: 'report', status: 'completed', total: 3, completed: 3, results: rows, report } }));
  await page.goto('/ui/');
  await page.locator('input[type=file]').setInputFiles({ name: 'goods.csv', mimeType: 'text/csv', buffer: Buffer.from('name\nавтомобиль\nБумага\nНеизвестный товар') });
  await expect(page.locator('tbody tr')).toHaveCount(3);
  await expect(page.locator('.summary-needs_review strong')).toHaveText('1');
  await page.locator('.summary-needs_review').click();
  await expect(page.locator('tbody tr')).toHaveCount(1);
  await expect(page.locator('tbody tr')).toContainText('Неоднозначное название');
  await expect(page.getByLabel('Сортировка')).toHaveValue('risk');
  await page.reload();
  await expect(page.locator('tbody tr')).toHaveCount(3);
  await expect(page.getByRole('status')).toContainText('Отчёт восстановлен с сервера');
  expect(uploads).toBe(1);
  const audit = await new AxeBuilder({ page }).analyze();
  expect(audit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
});

test('exception result never displays the ordinary prohibition verdict', async ({ page }) => {
  await page.route('**/classify', r => r.fulfill({ json: {
    ...result('Исключение', '26.20.11'), clause_ref: 'Приложение 1, позиция 1', code_role: 'Исключение', catalog_review_required: false,
  } }));
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Исключение');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.locator('.procurement strong')).toHaveText('Условия закупки требуют проверки');
  await expect(page.locator('.procurement')).toContainText('исключение из позиции');
});
