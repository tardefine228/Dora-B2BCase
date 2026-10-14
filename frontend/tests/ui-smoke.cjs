// Optional: npm exec --package=playwright -- playwright install chromium.
// Requires Playwright available to Node; API defaults to http://127.0.0.1:8000.
const { chromium } = require('playwright');
const { join } = require('node:path');
const { mkdirSync } = require('node:fs');
const base = process.env.BASE_URL || 'http://127.0.0.1:8000';
const output = process.env.SCREENSHOT_DIR || 'test-artifacts';
mkdirSync(output, {recursive:true});
(async () => {
  const browser = await chromium.launch({headless:true});
  const page = await browser.newPage({ viewport: {width:1280,height:900} });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${base}/ui/`);
  await page.getByLabel('Название товара', {exact:true}).fill('Поставка: Электрочайники');
  await page.getByRole('button',{name:'Определить код'}).click();
  await page.getByText('27.51.24.110',{exact:true}).first().waitFor();
  await page.screenshot({path:join(output,'ui-desktop.png'),fullPage:true});
  await page.locator('input[type=file]').setInputFiles({name:'products.csv',mimeType:'text/csv',buffer:Buffer.from('№;Наименование товара;Цена\n1;Электрочайники;2000\n2;Ноутбук Lenovo;50000')});
  await page.getByRole('button',{name:'Обработать список'}).click();
  await page.getByText('Обработка завершена',{exact:true}).waitFor();
  await page.getByRole('heading',{name:'Результаты списка'}).waitFor();
  const download = page.waitForEvent('download');
  await page.getByRole('button',{name:'Скачать Excel'}).click();
  await download;
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:join(output,'ui-mobile.png'),fullPage:true});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Mobile page overflows');
  await page.goto(`${base}/source/2-339-01?format=html`);
  await page.getByRole('heading',{name:'Электрочайники',exact:true}).waitFor();
  if(errors.length) throw new Error(errors.join('\n'));
  console.log('UI smoke passed: single, upload, batch, export, source, mobile; no JS errors.');
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
