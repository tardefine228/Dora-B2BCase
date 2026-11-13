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

test('ambiguous single query has no code and resolves after confirmation', async ({ page }) => {
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
  await page.locator('.clarification-choice').filter({ hasText: '29.10.2' }).click();
  await page.getByRole('button', { name: 'Подтвердить выбор' }).click();
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
  await row.locator('.clarification-choice').filter({ hasText: '29.10.2' }).click();
  await row.getByRole('button', { name: 'Подтвердить выбор' }).click();
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

test('long clarification titles stay readable inside a desktop table cell', async ({ page }, testInfo) => {
  const title = 'Микроскопы медицинские; микроскопы для клинической лабораторной диагностики; микроскопы биологические';
  const options = ['26.60.12.119', '26.70.22.150'].map((code, index) => ({
    option_id: `2-379-0${index + 1}`, code, title, confidence: .8,
    clause_ref: 'Приложение 2, позиция 379', clause_url: `/source/2-379-0${index + 1}`,
    code_role: 'Включён', catalog_review_required: false,
  }));
  const row = {
    ...pending(0), query: 'Микроскопы медицинские',
    clarification: { question: 'Уточните официальную позицию и код закупаемого товара.', options, catalog_sha256: 'a'.repeat(64), original },
  };
  await page.setViewportSize({ width: 1568, height: 1100 });
  await page.route('**/classify/file', r => r.fulfill({ json: { job_id: 'microscopes', total: 1, total_found: 1, filename: 'goods.csv', format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } }));
  await page.route('**/classify/batch/microscopes/events', r => r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' }));
  await page.route('**/classify/batch/microscopes', r => r.fulfill({ json: { job_id: 'microscopes', status: 'completed', total: 1, completed: 1, results: [row] } }));
  await page.goto('/ui/');
  await page.locator('input[type=file]').setInputFiles({ name: 'goods.csv', mimeType: 'text/csv', buffer: Buffer.from('name\nМикроскопы медицинские') });
  await page.getByText('Уточнить товар', { exact: true }).click();
  const card = page.locator('.compact-result .clarification-option');
  const choice = card.locator('.clarification-choice').first();
  const titleBox = await card.locator('.clarification-title').boundingBox();
  expect(titleBox).not.toBeNull();
  expect(titleBox!.width).toBeGreaterThan(180);
  await expect(card.locator('.clarification-code-row')).toHaveCount(2);
  expect(await card.locator('.clarification-title').count()).toBe(1);
  expect(await choice.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  expect((await choice.boundingBox())!.width).toBeGreaterThan(280);
  await expect(page.locator('.compact-result .clarification-clause').first()).toHaveText('Приложение 2, позиция 379');
  await testInfo.attach('microscopes-clarification', { body: await page.screenshot(), contentType: 'image/png' });
});

test('identical option titles expose their clause and structural code level', async ({ page }) => {
  const title = 'Автомобили-самосвалы, предназначенные для использования в условиях бездорожья';
  const options = [
    { option_id: '1-77-01', code: '28.92.29', clause_ref: 'Приложение 1, позиция 77' },
    { option_id: '3-175-01', code: '28.92.29.000', clause_ref: 'Приложение 3, позиция 175' },
  ].map(option => ({
    ...option, title, confidence: .9, clause_url: `/source/${option.option_id}`,
    code_role: 'Включён', catalog_review_required: false,
  }));
  await page.route('**/classify', r => r.fulfill({ json: {
    ...pending(), query: 'Самосвал',
    clarification: { question: 'Какой код и пункт требуются?', options, catalog_sha256: 'a'.repeat(64), original },
  } }));
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Самосвал');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.locator('.clarification-hierarchy')).toContainText('требуемый уровень ОКПД2');
  await expect(page.locator('.code-level').nth(0)).toHaveText('Укрупнённый код');
  await expect(page.locator('.code-level').nth(1)).toHaveText('Детальный код');
  await expect(page.locator('.clarification-clause').nth(0)).toHaveText('Приложение 1, позиция 77');
  await expect(page.locator('.clarification-clause').nth(1)).toHaveText('Приложение 3, позиция 175');
});

test('identical product titles become one card with nested code choices', async ({ page }) => {
  const repeated = [
    { option_id: '1-10-01', code: '26.51.42', clause_ref: 'Приложение 1, позиция 10', clause_url: '/source/1-10-01' },
    { option_id: '2-20-01', code: '26.51.42', clause_ref: 'Приложение 2, позиция 20', clause_url: '/source/2-20-01' },
    { option_id: '3-30-01', code: '26.51.43', clause_ref: 'Приложение 3, позиция 30', clause_url: '/source/3-30-01' },
  ].map(option => ({ ...option, title: 'Осциллографы', confidence: .8, code_role: 'Включён', catalog_review_required: false }));
  await page.route('**/classify', r => r.fulfill({ json: {
    ...pending(), query: 'Осциллограф', clarification: { question: 'Уточните код товара.', options: repeated, catalog_sha256: 'a'.repeat(64), original },
  } }));
  await page.route('**/classify/resolve', r => r.fulfill({ json: result('Осциллограф', '26.51.42') }));
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Осциллограф');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.locator('.clarification-option')).toHaveCount(1);
  await expect(page.locator('.clarification-code-row')).toHaveCount(2);
  await expect(page.locator('.clarification-title')).toHaveCount(1);
  await expect(page.locator('.clarification-option').first()).toContainText('Позиций в перечне: 2');
  await expect(page.locator('.clarification-warning')).toContainText('Не выбирайте наугад');
  await page.getByText('Позиции перечня (2)', { exact: true }).click();
  await expect(page.locator('.clarification-sources li')).toHaveCount(2);
  await expect(page.locator('.clarification-sources a')).toHaveCount(2);
});

test('file history keeps three reports and reopens a previous server result', async ({ page }) => {
  const jobs: Record<string, any> = {};
  let uploads = 0;
  await page.route('**/classify/file', r => {
    uploads++;
    const id = `history-${uploads}`;
    jobs[id] = { job_id: id, status: 'completed', total: 1, completed: 1,
      results: [{ ...result(`Товар ${uploads}`, '26.20.11'), item_id: 0 }] };
    return r.fulfill({ json: { job_id: id, total: 1, total_found: 1, filename: `${uploads}.csv`, format: 'csv', duplicates: 0, skipped: 0, truncated: false, warnings: [] } });
  });
  await page.route('**/classify/batch/*', r => {
    const url = new URL(r.request().url());
    if (url.pathname.endsWith('/events')) return r.fulfill({ contentType: 'text/event-stream', body: 'event: done\ndata: {}\n\n' });
    const id = url.pathname.split('/').at(-1)!;
    return r.fulfill({ json: jobs[id] });
  });
  await page.goto('/ui/');
  for (let index = 1; index <= 4; index++) {
    await page.locator('input[type=file]').setInputFiles({ name: `${index}.csv`, mimeType: 'text/csv', buffer: Buffer.from(`name\nТовар ${index}`) });
    await expect(page.locator('.history-file-copy strong').first()).toHaveText(`${index}.csv`);
  }
  await expect(page.locator('.history-list li')).toHaveCount(3);
  await expect(page.locator('.history-file-copy strong')).toHaveText(['4.csv', '3.csv', '2.csv']);
  const stored = await page.evaluate(() => localStorage.getItem('dora:file-history:v1')!);
  expect(stored).not.toContain('Товар');
  expect(stored).not.toContain('results');
  await page.getByRole('button', { name: /2\.csv/ }).click();
  await expect(page.locator('tbody')).toContainText('Товар 2');
  await expect(page.locator('tbody')).not.toContainText('Товар 4');
  await page.reload();
  await expect(page.locator('.history-list li')).toHaveCount(3);
  await expect(page.locator('tbody')).toContainText('Товар 2');
  const audit = await new AxeBuilder({ page }).analyze();
  expect(audit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
});

for (const width of [1440, 768, 390]) {
  test(`syringe guide and deliberate choice at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 });
    const options = ['32.50.13.110', '21.20.24.120', '32.50.50.190'].map((code, i) => ({
      ...clarificationOptions[0], code, option_id: `2-385-0${i + 1}`,
      okpd2_title: ['Шприцы, иглы, катетеры, канюли и аналогичные инструменты', 'Шовные материалы', 'Изделия медицинские, в том числе хирургические, прочие'][i],
      okpd2_source: `https://classifikators.ru/okpd/${code}`,
      title: 'Иглы хирургические; инструменты колющие; шприцы',
      clause_ref: 'Приложение 2, позиция 385', clause_url: `/source/2-385-0${i + 1}`,
    }));
    let saves = 0;
    await page.route('**/classify', r => r.fulfill({ json: {
      ...pending(), query: 'Шприц', clarification: { ...pending().clarification, question: 'Уточните назначение изделия', options },
    } }));
    await page.route('**/classify/resolve', r => {
      saves++;
      expect(r.request().postDataJSON().option_id).toBe('2-385-01');
      return r.fulfill({ json: { ...result('Шприц', '32.50.13.110'), decision_source: 'specialist' } });
    });
    await page.goto('/ui/');
    await page.getByLabel('Название товара', { exact: true }).fill('Шприц');
    await page.getByLabel('Название товара', { exact: true }).press('Enter');
    await expect(page.getByRole('button', { name: 'Подтвердить выбор' })).toBeDisabled();
    await expect(page.locator('.clarification-code-name')).toContainText([
      'Шприцы, иглы, катетеры', 'Шовные материалы', 'Изделия медицинские',
    ]);
    await page.getByRole('button', { name: 'Не знаю — оставить на проверку' }).click();
    await expect(page.locator('.selection-note')).toContainText('Код не назначен');
    expect(saves).toBe(0);
    await page.locator('.clarification-choice').filter({ hasText: '32.50.13.110' }).click();
    expect(saves).toBe(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.clarification').scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`syringe-${width}.png`), fullPage: true });
    const audit = await new AxeBuilder({ page }).analyze();
    expect(audit.violations.filter(v => ['critical', 'serious'].includes(v.impact ?? ''))).toEqual([]);
    await page.getByRole('button', { name: 'Подтвердить выбор' }).click();
    await expect(page.locator('.code')).toHaveText('32.50.13.110');
    expect(saves).toBe(1);
  });
}

test('butter uses reference names and marks the broader category', async ({ page }) => {
  const options = [
    { code: '10.51.30.100', okpd2_title: 'Масло сливочное, в том числе с пищевкусовыми компонентами' },
    { code: '10.51.30.110', okpd2_title: 'Масло сливочное' },
  ].map((item, i) => ({ ...clarificationOptions[0], ...item, title: 'Масло сливочное', option_id: `2-${450+i}-01` }));
  await page.route('**/classify', r => r.fulfill({ json: { ...pending(), query: 'Масло сливочное',
    clarification: { ...pending().clarification, options } } }));
  await page.goto('/ui/');
  await page.getByLabel('Название товара', { exact: true }).fill('Масло сливочное');
  await page.getByLabel('Название товара', { exact: true }).press('Enter');
  await expect(page.locator('.clarification-code-name mark')).toContainText(['пищевкусовыми', 'компонентами']);
  await expect(page.locator('.code-level')).toHaveText(['Укрупнённый код', 'Детальный код']);
  await expect(page.locator('.choice-hint').first()).toContainText('Общая категория');
  await expect(page.getByRole('button', { name: 'Подтвердить выбор' })).toBeDisabled();
});
