import { useEffect, useRef, useState } from 'react';
import { api, followBatch, type Job, type Result, type UploadAccepted, type Verdict } from './api';
import { exportResults, exportResultsCsv } from './files';
import { STATUS_HELP, STATUS_LABEL, codeLabel, duplicateAppendixHint, notFoundHint, verifierNote } from './status';
import { orderResults, type SortMode } from './rows';
import { storage } from './storage';
import { isRunningJob, navigationType, parseJobMarker, readStoredView, type StoredView } from './view';

const ACTIVE_JOB_KEY = 'dora:active-job:v1';
// Ключи удалённой локальной истории: их надо стирать у тех, кто успел включить
// галочку, иначе названия товаров останутся в браузере без всякого UI.
const LEGACY_HISTORY_KEYS = ['dora:local-history:v1', 'dora:history-enabled:v1'];
const message = (error: unknown) => error instanceof Error ? error.message : 'Ошибка. Попробуйте ещё раз.';
const percent = (value: number | null) => value === null ? '—' : `${Math.round(value * 100)}%`;
const isSafeUrl = (url: string | null): url is string => !!url && (/^\/source\/[\w-]+$/.test(url) || /^https?:\/\//i.test(url));

function Icon({ kind = 'search' }: { kind?: 'search' | 'upload' }) {
  return <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
    {kind === 'search' ? <><circle cx="10" cy="10" r="6.5"/><path d="m15 15 6 6"/></> :
      <><path d="M14 3H5v18h14V8zM14 3v5h5M12 17V10m-4 4 4-4 4 4"/></>}
  </svg>;
}
function SourceLink({ url }: { url: string | null }) {
  return isSafeUrl(url) ? <a href={url.startsWith("/source/") ? `${url}?view=clean-2` : url}>Открыть пункт постановления →</a>
    : <span className="muted">Ссылка на пункт недоступна</span>;
}
function Alternatives({ result }: { result: Result }) {
  if (!result.alternatives.length) return null;
  return <details className="details">
    <summary>Альтернативные варианты ({result.alternatives.length})</summary>
    <div className="alternative-list">{result.alternatives.map((alt, i) =>
      <div className="alternative" key={`${alt.code ?? 'unknown'}-${i}`}>
        <strong>{alt.code ?? 'Код не определён'}</strong>
        <span>{alt.title ?? 'Наименование не указано'}</span>
        <small>Оценка соответствия: {percent(alt.confidence)} — не откалиброванная вероятность.</small>
        {alt.clauseRef && <small>{alt.clauseRef}</small>}
        <SourceLink url={alt.url}/>
      </div>)}</div>
  </details>;
}
function Feedback({ result, onSaved }: { result: Result; onSaved: (r: Result, verdict: Verdict, corrected: string | null) => void }) {
  const [choice, setChoice] = useState<Verdict | null>(null);
  const [corrected, setCorrected] = useState('');
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(!!result.userVerdict);
  const [error, setError] = useState('');
  const submit = async (verdict: Verdict) => {
    setChoice(verdict); setSent(false); setError('');
    if (verdict === 'incorrect' && !corrected.trim()) return;
    setSending(true);
    try { await api.feedback(result, verdict, corrected); setSent(true); onSaved(result, verdict, verdict === 'incorrect' ? corrected.trim() : null); }
    catch (e) { setError(message(e)); }
    finally { setSending(false); }
  };
  return <div className="feedback">
    <strong>Помогите улучшить качество классификации</strong>
    <div className="feedback-actions" aria-label="Оценить результат">
      <button type="button" className="outline small" disabled={sending || sent || !result.code} onClick={() => void submit('correct')}>✓ Код верный</button>
      <button type="button" className="outline small" disabled={sending || sent} onClick={() => { setChoice('incorrect'); setError(''); }}>✎ Код неверный</button>
      <button type="button" className="outline small" disabled={sending || sent} onClick={() => void submit('absent')}>Нет в перечне</button>
    </div>
    {choice === 'incorrect' && !sent && <form className="correction" onSubmit={e => { e.preventDefault(); void submit('incorrect'); }}>
      <label>Правильный код ОКПД2
        <input value={corrected} onChange={e => setCorrected(e.target.value)} maxLength={32} placeholder="Например, 27.51.24.110" required disabled={sending}/>
      </label>
      <button type="submit" className="primary small" disabled={sending || !corrected.trim()}>{sending ? 'Сохраняем…' : 'Отправить исправление'}</button>
    </form>}
    {error && <p className="field-error" role="alert">{error}</p>}
    {sent && <p className="field-success" role="status">Спасибо! Ваша оценка сохранена на сервере.</p>}
  </div>;
}
function ResultCard({ result, compact = false, onSaved }: { result: Result; compact?: boolean; onSaved: (r: Result, verdict: Verdict, corrected: string | null) => void }) {
  const missingCode = result.status !== 'not_found' && !result.code;
  // «Нет в перечне» и «нашёл, но не уверен» — разные ответы, и подсказка у них
  // тоже разная: в первом случае важно, есть ли вообще похожие позиции.
  const hint = result.status === 'not_found' ? notFoundHint(result) : STATUS_HELP[result.status];
  const twin = duplicateAppendixHint(result);
  const verified = verifierNote(result);
  return <div className={compact ? 'compact-result' : ''}>
    {compact ? <>
      <div className="nowrap">{result.code ?? '—'}</div><p>{result.title ?? "Название категории не указано"}</p><small>{STATUS_LABEL[result.status]}</small>
    </> : <>
      <div className="result-heading"><h2>Результат</h2><span className={`badge status-${result.status}`}>{STATUS_LABEL[result.status]}</span></div>
      <div className="result-grid">
        <div><p className="muted">{codeLabel(result.status)}</p>
          {result.status === 'not_found' ? <p className="title">Товар не найден в перечне</p>
            : missingCode ? <p className="title">Сервис не вернул код — требуется проверка</p>
              : <><div className="code">{result.code}</div><p className="title">Категория: {result.title ?? "название не указано"}</p>
                {result.clauseRef && <p className="muted">{result.clauseRef}</p>}</>}
        </div>
        <div className="confidence"><strong>{percent(result.confidence)}</strong><p>Уверенность модели</p>
          {result.confidence !== null && <progress max={1} value={result.confidence} aria-label="Уверенность модели"/>}</div>
      </div>
    </>}
    <p className="status-help">{missingCode ? 'Без кода нельзя считать результат уверенным.' : hint}</p>
    {verified && <p className="status-help" role="note">{verified}</p>}
    {twin && <p className="status-help" role="note">{twin}</p>}
    {!compact && <div className="source"><SourceLink url={result.url}/></div>}
    <Alternatives result={result}/>
    <details className="details"><summary>Сообщить об ошибке (для экспертов)</summary><Feedback key={`${result.name}-${result.code}`} result={result} onSaved={onSaved}/></details>
  </div>;
}

function readSession(): StoredView {
  // Просмотр возвращается только по «Назад» (ушли в пункт перечня и вернулись).
  // При обычном открытии сайта запись удаляется: см. src/view.ts.
  try { return readStoredView(sessionStorage, navigationType()) ?? {}; }
  catch { return {}; }
}
const savedView = readSession();

export default function App() {
  const [name, setName] = useState<string>(savedView.name ?? '');
  const [busy, setBusy] = useState<'single' | 'file' | 'batch' | null>(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Result | null>(savedView.result ?? null);
  const [filename, setFilename] = useState<string>(savedView.filename ?? '');
  const [fileInfo, setFileInfo] = useState<UploadAccepted | null>(savedView.fileInfo ?? null);
  const [progress, setProgress] = useState<{ processed: number; total: number; state: 'running' | 'done' | 'failed' } | null>(savedView.progress ?? null);
  const [results, setResults] = useState<Result[]>(savedView.results ?? []);
  const [sort, setSort] = useState<SortMode>('original');
  const [drag, setDrag] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [storageWarning, setStorageWarning] = useState('');
  // Одноразовая строка о том, что прошлая обработка не восстановлена.
  const [notice, setNotice] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const locked = useRef(false);

  useEffect(() => {
    // Локальная история проверок удалена из интерфейса (она не была нужна для
    // работы и хранила названия товаров). Заодно вычищаем записи у тех, кто её
    // включал: без UI они остались бы в браузере навсегда.
    for (const key of LEGACY_HISTORY_KEYS) storage.removeItem(key);
  }, []);

  useEffect(() => {
    try { sessionStorage.setItem('dora:current-view:v1', JSON.stringify({ name, result, filename, fileInfo, progress, results })); }
    catch { setStorageWarning('Не удалось сохранить текущие результаты для возврата по «Назад». Скачайте CSV перед переходом по ссылке.'); }
  }, [name, result, filename, fileInfo, progress, results]);

  function saveVerdict(r: Result, verdict: Verdict, corrected: string | null) {
    const mark = (item: Result) => item.name === r.name && item.code === r.code
      ? { ...item, userVerdict: verdict, correctedCode: corrected } : item;
    setResult(old => old ? mark(old) : old);
    setResults(old => old.map(mark));
  }
  async function follow(jobId: string, total: number) {
    abort.current?.abort();
    const controller = new AbortController(); abort.current = controller;
    locked.current = true; setBusy('batch'); setError('');
    setProgress({ processed: 0, total, state: 'running' });
    try {
      const done = await followBatch(jobId, job => {
        if (controller.signal.aborted) return;
        setProgress({ processed: job.processed, total: job.total,
          state: job.status === 'completed' ? 'done' : job.status === 'failed' ? 'failed' : 'running' });
        if (job.results.length) setResults(job.results);
      }, controller.signal);
      if (controller.signal.aborted) return;
      setResults(done.results);
      if (done.status !== 'completed') {
        setError(done.error ?? 'Не удалось обработать все товары. Частичные результаты сохранены.');
      }
      storage.removeItem(ACTIVE_JOB_KEY);
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(message(e));
        setProgress(old => old ? { ...old, state: 'failed' } : old);
      }
      // Keep job ID on timeout/connection failure: user may restore the batch.
    } finally {
      if (!controller.signal.aborted) { locked.current = false; setBusy(null); }
    }
  }
  useEffect(() => {
    void resumeActiveJob();
    return () => { abort.current?.abort(); locked.current = false; };
  }, []);

  /**
   * Возобновляет только *идущую* обработку.
   *
   * Задание лежит в localStorage, чтобы пережить закрытие вкладки, но показывать
   * результаты задания, которое завершилось в прошлой сессии, нельзя: страница
   * открывается пустой, а о потере результатов честно сообщает строка ниже.
   */
  async function resumeActiveJob() {
    const marker = parseJobMarker(storage.getItem(ACTIVE_JOB_KEY));
    if (!marker) { storage.removeItem(ACTIVE_JOB_KEY); return; }
    let job: Job;
    try { job = await api.getBatch(marker.id); }
    catch { return; }  // Сервер недоступен — не пугаем и не теряем задание.
    if (!isRunningJob(job.status)) {
      storage.removeItem(ACTIVE_JOB_KEY);
      const name = marker.filename ? `«${marker.filename}»` : 'файла';
      setNotice(`Обработка ${name} завершилась, пока страница была закрыта. Результаты не восстановлены — загрузите файл снова, если они нужны.`);
      return;
    }
    if (marker.filename) setFilename(marker.filename);
    await follow(marker.id, job.total || marker.total);
  }

  async function single() {
    if (locked.current) return;
    const value = name.trim();
    if (!value) { setError('Введите название товара.'); return; }
    locked.current = true; setBusy('single'); setError(''); setNotice(''); setResult(null); setResults([]); setProgress(null);
    try { const found = await api.classify(value); setResult(found); }
    catch (e) { setError(message(e)); }
    finally { locked.current = false; setBusy(null); }
  }
  async function load(file?: File) {
    if (!file || locked.current) return;
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!['csv','tsv','xlsx','xlsm','pdf','docx','json','txt','md'].includes(ext)) {
      setError('Поддерживаются Excel (.xlsx), CSV, PDF, DOCX, JSON и текстовые файлы.'); return;
    }
    if (file.size > 9_500_000) { setError('Файл должен быть меньше 9,5 МБ (ограничение на загрузку).'); return; }
    locked.current = true; setBusy('file'); setError(''); setNotice(''); setFileInfo(null); setFilename(''); setResults([]); setProgress(null); setResult(null);
    try {
      const accepted = await api.uploadFile(file);
      setFilename(file.name); setFileInfo(accepted);
      if (!storage.setItem(ACTIVE_JOB_KEY, JSON.stringify({ id: accepted.job_id, total: accepted.total, filename: file.name }))) {
        setStorageWarning('Браузер не сохранил задание. Дождитесь завершения обработки, не закрывая страницу.');
      }
      locked.current = false;
      await follow(accepted.job_id, accepted.total);
    } catch (e) { setError(message(e)); }
    finally { locked.current = false; setBusy(null); if (input.current) input.current.value = ''; }
  }
  // Порядок строк: файл, название или уверенность (см. src/rows.ts). Номер в
  // первой колонке — просто счётчик строк этого порядка, 1…N.
  const sorted = orderResults(results, sort);

  return <>
    <header><div className="brand"><span className="brand-icon"><Icon/></span>Классификатор товаров</div></header>
    <main>
      <section className="intro"><h1>Найдите код товара</h1><p>Поиск позиции в перечне ПП № 1875 по названию товара</p></section>
      <div className="forms">
        <form className="card single" onSubmit={e => { e.preventDefault(); void single(); }}>
          <h2>Один товар</h2><label htmlFor="product">Название товара</label>
          <div className="input-wrap"><input id="product" value={name} maxLength={500} disabled={!!busy} placeholder="Ноутбук Lenovo IdeaPad 15.6 дюймов"
            onChange={e => { setName(e.target.value); setResult(null); }}/>
            <button type="button" className="clear" aria-label="Очистить название" disabled={!!busy || !name}
              onClick={() => { setName(''); setResult(null); }}>×</button></div>
          <p className="hint">Укажите тип товара, модель и основные характеристики</p>
          <button type="submit" className="primary submit" disabled={!!busy}>
            {busy === 'single' ? 'Определяем код…' : 'Определить код'} <span aria-hidden="true">→</span>
          </button>
        </form>
        <section className="card"><h2>Список товаров</h2>
          <div className={`drop ${drag ? 'drag' : ''}`}
            onDragOver={e => { e.preventDefault(); if (!busy) setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); if (busy) return;
              if (e.dataTransfer.files.length !== 1) { setError('Загрузите один файл за раз.'); return; }
              void load(e.dataTransfer.files[0]); }}>
            <Icon kind="upload"/><strong>{busy === 'file' ? 'Загружаем файл…' : filename || 'Перетащите файл сюда'}</strong>
            <span>Excel, CSV, PDF, DOCX, JSON · до 100 товаров</span>
          </div>
          <input ref={input} type="file" accept=".xlsx,.xlsm,.csv,.tsv,.pdf,.docx,.json,.txt,.md" hidden aria-label="Выбрать файл товаров"
            onChange={e => void load(e.target.files?.[0])}/>
          <button type="button" className="outline wide" disabled={!!busy} onClick={() => input.current?.click()}>
            Выбрать файл
          </button>
          {fileInfo && <div className="file-options" role="status">
            <p><strong>Извлечено: {fileInfo.total} товаров</strong></p>
            <p className="hint">{fileInfo.source_column ? `Столбец: ${fileInfo.source_column}. ` : ''}Пропущено: {fileInfo.skipped}. Дубликатов: {fileInfo.duplicates}.</p>
            {fileInfo.truncated && <p className="field-error">Список обрезан до 100 позиций.</p>}
            {fileInfo.warnings.map((w,i) => <p className="hint" key={i}>{w}</p>)}
          </div>}
        </section>
      </div>
      {error && <div className="error" role="alert">{error}</div>}
      {notice && <p className="hint" role="status">{notice}</p>}
      {storageWarning && <p className="error" role="status">{storageWarning}</p>}
      <div aria-live="polite" aria-busy={!!busy}>
        {progress && <section className="card progress" aria-label="Прогресс обработки">
          <strong>{progress.state === 'running' ? 'Обрабатываем список…' : progress.state === 'done' ? 'Обработка завершена' : 'Обработка остановлена'}</strong>
          <span>{progress.processed} из {progress.total}</span>
          <progress max={progress.total} value={progress.processed} aria-label="Обработано товаров"/>
        </section>}
        {result && <section className="card result"><ResultCard result={result} onSaved={saveVerdict}/></section>}
        {results.length > 0 && <section className="card result">
          <div className="result-heading"><h2>Результаты списка</h2>
          </div>
          <div className="toolbar"><label htmlFor="sort">Сортировка
            <select id="sort" value={sort} onChange={e => setSort(e.target.value as SortMode)}>
              <option value="original">Как в файле</option><option value="name">По названию</option><option value="confidence">По уверенности</option>
            </select></label>
            <span className="toolbar-actions">
              <button type="button" className="outline" disabled={exporting} onClick={async () => { setExporting(true);
                try { await exportResults(sorted); } catch (e) { setError(message(e)); } finally { setExporting(false); }
              }}>{exporting ? 'Сохраняем…' : 'Скачать Excel'}</button>
              <button type="button" className="outline" disabled={exporting} onClick={() => {
                try { exportResultsCsv(sorted); } catch (e) { setError(message(e)); }
              }}>Скачать CSV</button>
            </span>
          </div>
          <div className="table-scroll"><table><thead><tr><th scope="col" className="row-num">№</th><th scope="col">Товар</th><th scope="col">Код и статус</th><th scope="col">Уверенность</th><th scope="col">Подробности</th></tr></thead>
            <tbody>{sorted.map((r, index) => <tr key={results.indexOf(r)}>
              <td className="row-num">{index + 1}</td>
              <td>{r.name}</td><td>{r.code ?? '—'}{r.code && <span className="cell-status">Категория: {r.title ?? 'название не указано'}</span>}<span className="muted cell-status">{STATUS_LABEL[r.status]}</span></td>
              <td>{percent(r.confidence)}</td><td><SourceLink url={r.url}/>
                <details className="details"><summary>Проверить результат</summary><ResultCard result={r} compact onSaved={saveVerdict}/></details></td>
            </tr>)}</tbody></table></div>
        </section>}
      </div>
      <footer>Подбор кода — подсказка, а не юридический вердикт.<br/>Результат требует проверки специалистом.</footer>
    </main>
  </>;
}
