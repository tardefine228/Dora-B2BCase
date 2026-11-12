import { useEffect, useRef, useState } from 'react';
import { api, followBatch, type ClarificationOption, type Job, type ProcurementReport, type ReportBucket, type ReportReasonCode, type Result, type UploadAccepted, type Verdict } from './api';
import { exportResults, exportResultsCsv } from './files';
import { procurementVerdict, PROCUREMENT_SCOPE, PROCUREMENT_SOURCE } from './procurement';
import { STATUS_HELP, STATUS_LABEL, codeLabel, duplicateAppendixHint, notFoundHint, verifierNote } from './status';
import { orderResults, type SortMode } from './rows';
import { storage } from './storage';
import { isRunningJob, navigationType, parseJobMarker, readStoredView, type StoredView } from './view';
import { BUCKET_LABEL, filterReportResults, REASON_LABEL, reportItem } from './report';
import { clarificationCodeLevel, CODE_LEVEL_LABEL, groupClarificationOptions } from './clarification';
import { addFileHistory, FILE_HISTORY_KEY, parseFileHistory, updateFileHistoryStatus, type FileHistoryEntry } from './history';

const ACTIVE_JOB_KEY = 'dora:active-job:v1';
const PRODUCT_NAME_MAX_LENGTH = 2000;
// Ключи удалённой локальной истории: их надо стирать у тех, кто успел включить
// галочку, иначе названия товаров останутся в браузере без всякого UI.
const LEGACY_HISTORY_KEYS = ['dora:local-history:v1', 'dora:history-enabled:v1'];
const message = (error: unknown) => error instanceof Error ? error.message : 'Ошибка. Попробуйте ещё раз.';
const percent = (value: number | null) => value === null ? '—' : `${Math.round(value * 100)}%`;
const isSafeUrl = (url: string | null): url is string => !!url && (/^\/source\/[\w-]+$/.test(url) || /^https?:\/\//i.test(url));
const FILE_STATUS_LABEL: Record<Job['status'], string> = {
  queued: 'В очереди', running: 'Обрабатывается', completed: 'Готово', failed: 'С ошибкой',
};
function historyDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Дата не указана'
    : new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }).format(date);
}

function UploadIcon() {
  return <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
    <path d="M14 3H5v18h14V8zM14 3v5h5M12 17V10m-4 4 4-4 4 4"/>
  </svg>;
}
function SourceLink({ url }: { url: string | null }) {
  return isSafeUrl(url) ? <a className="source-link" href={url.startsWith("/source/") ? `${url}?view=clean-2` : url}><span>Открыть пункт постановления</span><span className="source-link-arrow" aria-hidden="true">→</span></a>
    : <span className="muted">Ссылка на пункт недоступна</span>;
}
function SourcePositions({ options }: { options: ClarificationOption[] }) {
  const positions = options.flatMap(option =>
    (option.clauseRef ?? '').split('; ').map(ref => ref.trim()).filter(Boolean)
      .map((clauseRef, index) => ({ clauseRef, url: index === 0 ? option.url : null, optionId: option.optionId })));
  if (positions.length <= 1) return <SourceLink url={options[0]?.url ?? null}/>;
  return <details className="clarification-sources">
    <summary>Позиции перечня ({positions.length})</summary>
    <ul>{positions.map(position => <li key={`${position.optionId}-${position.clauseRef}`}>
      <span>{position.clauseRef}</span>{position.url && <SourceLink url={position.url}/>}
    </li>)}</ul>
  </details>;
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
function ClarificationBox({ result, onResolve, onResolved }: {
  result: Result;
  onResolve: (result: Result, optionId: string) => Promise<Result>;
  onResolved: (original: Result, resolved: Result) => void;
}) {
  const [sending, setSending] = useState<string | null>(null);
  const [error, setError] = useState('');
  const clarification = result.clarification;
  if (!clarification) return null;
  const hasCodeHierarchy = clarification.options.some(option =>
    clarificationCodeLevel(option, clarification.options) !== null);
  const choose = async (optionId: string) => {
    if (sending) return;
    setSending(optionId); setError('');
    try { onResolved(result, await onResolve(result, optionId)); }
    catch (e) { setError(message(e)); }
    finally { setSending(null); }
  };
  return <section className="clarification" aria-labelledby={`question-${result.itemId ?? 'single'}`}>
    <h3 id={`question-${result.itemId ?? 'single'}`}>{clarification.question}</h3>
    <p>Выберите один вариант. До выбора код ОКПД2 не назначается.</p>
    {hasCodeHierarchy && <p className="clarification-hierarchy">Укрупнённый и детальный код
      могут относиться к одному товару. Выбирайте требуемый уровень ОКПД2 и приложение перечня.</p>}
    <div className="clarification-options">{groupClarificationOptions(clarification.options).map(group => {
      const option = group.options[0];
      const level = clarificationCodeLevel(option, clarification.options);
      const review = group.options.some(item => item.catalogReviewRequired);
      return <div className="clarification-option" key={group.key}>
        <button type="button" className="clarification-choice" disabled={sending !== null}
          onClick={() => void choose(option.optionId)}>
          <strong>{group.code}{level &&
            <span className="code-level">{CODE_LEVEL_LABEL[level]}</span>}</strong>
          <span className="clarification-title">{group.title}</span>
          {group.clauseRefs.length === 1 &&
            <small className="clarification-clause">{group.clauseRefs[0]}</small>}
          {group.clauseRefs.length > 1 &&
            <small className="clarification-clause">Найден в {group.clauseRefs.length} позициях перечня</small>}
          <small>{sending === option.optionId ? 'Сохраняем выбор…' : 'Выбрать этот код'}
            {review ? ' · потребуется сверка каталога' : ''}</small>
        </button>
        <SourcePositions options={group.options}/>
      </div>;
    })}
    </div>
    {error && <p className="field-error" role="alert">{error} Обновите результат и попробуйте снова.</p>}
  </section>;
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
function resultStatus(result: Result) {
  if (result.decisionSource === 'specialist' && result.status === 'ok') return 'Выбрано специалистом';
  return procurementVerdict(result).caution ?? STATUS_LABEL[result.status];
}
function Procurement({ result, brief = false }: { result: Result; brief?: boolean }) {
  const verdict = procurementVerdict(result);
  return <div className={`procurement procurement-${verdict.tone}${brief ? ' procurement-brief' : ''}`}>
    <strong>{verdict.title}</strong>
    <p>{verdict.explanation}</p>
  </div>;
}
function ResultCard({ result, compact = false, onSaved, onResolve, onResolved }: {
  result: Result;
  compact?: boolean;
  onSaved: (r: Result, verdict: Verdict, corrected: string | null) => void;
  onResolve: (result: Result, optionId: string) => Promise<Result>;
  onResolved: (original: Result, resolved: Result) => void;
}) {
  const missingCode = (result.status === 'ok' || result.status === 'review') && !result.code;
  // «Нет в перечне» и «нашёл, но не уверен» — разные ответы, и подсказка у них
  // тоже разная: в первом случае важно, есть ли вообще похожие позиции.
  const hint = result.status === 'not_found' ? notFoundHint(result) : STATUS_HELP[result.status];
  const twin = duplicateAppendixHint(result);
  const verified = verifierNote(result);
  return <div className={compact ? 'compact-result' : ''}>
    {compact ? <>
      <div className="nowrap">{result.code ?? '—'}</div><p>{result.title ?? "Название категории не указано"}</p><small>{resultStatus(result)}</small>
    </> : <>
      <div className="result-heading"><h2>Результат</h2><span className={`badge status-${result.status}`}>{resultStatus(result)}</span></div>
      <div className="result-grid">
        <div><p className="muted">{codeLabel(result.status)}</p>
          {result.status === 'not_found' ? <p className="title">Совпадение не найдено</p>
            : result.status === 'needs_clarification' ? <p className="title">Код не назначен до уточнения</p>
              : missingCode ? <p className="title">Сервис не вернул код — требуется проверка</p>
                : <><div className="code">{result.code}</div><p className="title">Категория: {result.title ?? "название не указано"}</p>
                {result.clauseRef && <p className="muted">{result.clauseRef}</p>}</>}
        </div>
        {result.decisionSource === 'specialist'
          ? <div className="confidence"><strong>Ручной выбор</strong><p>Оценка модели не показывается</p></div>
          : <div className="confidence"><strong>{percent(result.confidence)}</strong><p>Уверенность в подборе кода</p>
            {result.confidence !== null && <progress max={1} value={result.confidence} aria-label="Уверенность модели"/>}</div>}
      </div>
    </>}
    {result.status === 'needs_clarification' &&
      <ClarificationBox result={result} onResolve={onResolve} onResolved={onResolved}/>}
    {result.decisionSource === 'specialist' && result.resolution && <p className="resolution-note" role="status">
      Выбрано специалистом. Исходный ответ модели: {result.resolution.original.code ?? 'без кода'} · {STATUS_LABEL[result.resolution.original.status]}.
    </p>}
    <Procurement result={result}/>
    <p className="status-help">{missingCode ? 'Без кода нельзя считать результат уверенным.' : hint}{result.status === 'not_found' && ' Процент показывает уверенность в подборе кода, а не вероятность отсутствия товара в постановлении.'}</p>
    {verified && <p className="status-help" role="note">{verified}</p>}
    {twin && <p className="status-help" role="note">{twin}</p>}
    {!compact && <div className="source"><SourceLink url={result.url}/></div>}
    <Alternatives result={result}/>
    {result.status !== 'needs_clarification' && <details className="details"><summary>Сообщить об ошибке (для экспертов)</summary><Feedback key={`${result.name}-${result.code}`} result={result} onSaved={onSaved}/></details>}
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
  const [jobId, setJobId] = useState<string | null>(savedView.jobId ?? null);
  const [report, setReport] = useState<ProcurementReport | null>(null);
  const [bucketFilter, setBucketFilter] = useState<ReportBucket | 'all'>('all');
  const [reasonFilter, setReasonFilter] = useState<ReportReasonCode | 'all'>('all');
  const [sort, setSort] = useState<SortMode>('original');
  const [drag, setDrag] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [storageWarning, setStorageWarning] = useState('');
  const [fileHistory, setFileHistory] = useState<FileHistoryEntry[]>(() =>
    parseFileHistory(storage.getItem(FILE_HISTORY_KEY)));
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
    try { sessionStorage.setItem('dora:current-view:v1', JSON.stringify({ name, result, filename, fileInfo, progress, results, jobId })); }
    catch { setStorageWarning('Не удалось сохранить текущие результаты для возврата по «Назад». Скачайте CSV перед переходом по ссылке.'); }
  }, [name, result, filename, fileInfo, progress, results, jobId]);

  useEffect(() => {
    storage.setItem(FILE_HISTORY_KEY, JSON.stringify(fileHistory));
  }, [fileHistory]);

  function markHistory(id: string, status: Job['status']) {
    setFileHistory(current => updateFileHistoryStatus(current, id, status));
  }
  async function refreshFileHistory() {
    const snapshot = parseFileHistory(storage.getItem(FILE_HISTORY_KEY));
    await Promise.all(snapshot.map(async entry => {
      try {
        const job = await api.getBatch(entry.jobId);
        setFileHistory(current => updateFileHistoryStatus(current, entry.jobId, job.status));
      } catch { /* Keep the metadata: a temporary network error must not erase history. */ }
    }));
  }

  function saveVerdict(r: Result, verdict: Verdict, corrected: string | null) {
    const mark = (item: Result) => item.name === r.name && item.code === r.code
      ? { ...item, userVerdict: verdict, correctedCode: corrected } : item;
    setResult(old => old ? mark(old) : old);
    setResults(old => old.map(mark));
  }
  async function resolveClarification(original: Result, optionId: string): Promise<Result> {
    if (jobId) {
      const itemId = original.itemId ?? results.indexOf(original);
      if (itemId < 0) throw new Error('Позиция не найдена в текущем списке. Загрузите файл повторно.');
      return api.resolveBatch(jobId, itemId, original, optionId);
    }
    return api.resolve(original, optionId);
  }
  function clarificationResolved(original: Result, resolved: Result) {
    setResult(current => current === original ? resolved : current);
    setResults(current => current.map(item => item === original ? resolved : item));
    setError('');
    if (jobId) {
      void api.getBatch(jobId).then(job => setReport(job.report)).catch((error: unknown) => {
        setError(`Выбор сохранён, но сводку не удалось обновить: ${message(error)}`);
      });
    }
  }
  async function follow(jobId: string, total: number) {
    setJobId(jobId);
    markHistory(jobId, 'running');
    abort.current?.abort();
    const controller = new AbortController(); abort.current = controller;
    locked.current = true; setBusy('batch'); setError('');
    setProgress({ processed: 0, total, state: 'running' });
    try {
      const done = await followBatch(jobId, job => {
        if (controller.signal.aborted) return;
        markHistory(jobId, job.status);
        setProgress({ processed: job.processed, total: job.total,
          state: job.status === 'completed' ? 'done' : job.status === 'failed' ? 'failed' : 'running' });
        if (job.results.length) setResults(job.results);
        if (job.report) setReport(job.report);
      }, controller.signal);
      if (controller.signal.aborted) return;
      markHistory(jobId, done.status);
      setResults(done.results);
      setReport(done.report);
      if (done.status !== 'completed') {
        setError(done.error ?? 'Не удалось обработать все товары. Частичные результаты сохранены.');
      }
      // Keep only the server job id: a completed report is restored after reload.
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
    void refreshFileHistory();
    return () => { abort.current?.abort(); locked.current = false; };
  }, []);

  /** Restore running work or a completed server report using only its opaque id. */
  async function resumeActiveJob() {
    const marker = parseJobMarker(storage.getItem(ACTIVE_JOB_KEY));
    if (!marker) { storage.removeItem(ACTIVE_JOB_KEY); return; }
    let job: Job;
    try { job = await api.getBatch(marker.id); }
    catch (error) {
      setNotice(`Отчёт не восстановлен: ${message(error)} Идентификатор сохранён для повторной попытки.`);
      return;
    }
    if (marker.filename) setFilename(marker.filename);
    if (isRunningJob(job.status)) {
      await follow(marker.id, job.total || marker.total);
      return;
    }
    setJobId(job.job_id);
    setResults(job.results);
    setReport(job.report);
    setProgress({
      processed: job.processed, total: job.total,
      state: job.status === 'completed' ? 'done' : 'failed',
    });
    if (job.status === 'completed') setNotice('Отчёт восстановлен с сервера. Срок хранения ограничен настройкой JOB_TTL_HOURS.');
    else setError(job.error ?? 'Сохранённая обработка завершилась с ошибкой.');
  }

  async function single() {
    if (locked.current) return;
    const value = name.trim();
    if (!value) { setError('Введите название товара.'); return; }
    locked.current = true; setBusy('single'); setError(''); setNotice(''); setResult(null); setResults([]); setProgress(null); setJobId(null); setReport(null);
    storage.removeItem(ACTIVE_JOB_KEY);
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
    locked.current = true; setBusy('file'); setError(''); setNotice(''); setFileInfo(null); setFilename(''); setResults([]); setProgress(null); setResult(null); setJobId(null); setReport(null); setBucketFilter('all'); setReasonFilter('all');
    storage.removeItem(ACTIVE_JOB_KEY);
    try {
      const accepted = await api.uploadFile(file);
      setFilename(file.name); setFileInfo(accepted);
      setFileHistory(current => addFileHistory(current, {
        jobId: accepted.job_id, filename: file.name, total: accepted.total,
        createdAt: new Date().toISOString(), status: 'queued', fileInfo: accepted,
      }));
      if (!storage.setItem(ACTIVE_JOB_KEY, JSON.stringify({ id: accepted.job_id, total: accepted.total, filename: file.name }))) {
        setStorageWarning('Браузер не сохранил задание. Дождитесь завершения обработки, не закрывая страницу.');
      }
      locked.current = false;
      await follow(accepted.job_id, accepted.total);
    } catch (e) { setError(message(e)); }
    finally { locked.current = false; setBusy(null); if (input.current) input.current.value = ''; }
  }
  async function openHistory(entry: FileHistoryEntry) {
    if (locked.current) return;
    locked.current = true; setBusy('batch'); setError(''); setNotice('');
    setResult(null); setResults([]); setReport(null); setProgress(null); setJobId(null);
    setFilename(entry.filename); setFileInfo(entry.fileInfo);
    setBucketFilter('all'); setReasonFilter('all');
    try {
      const job = await api.getBatch(entry.jobId);
      setJobId(job.job_id);
      storage.setItem(ACTIVE_JOB_KEY, JSON.stringify({
        id: job.job_id, total: job.total || entry.total, filename: entry.filename,
      }));
      markHistory(job.job_id, job.status);
      if (isRunningJob(job.status)) {
        await follow(job.job_id, job.total || entry.total);
      } else {
        setResults(job.results); setReport(job.report);
        setProgress({ processed: job.processed, total: job.total,
          state: job.status === 'completed' ? 'done' : 'failed' });
        if (job.status === 'completed') setNotice(`Открыт отчёт «${entry.filename}». Результаты загружены с сервера.`);
        else setError(job.error ?? 'Обработка этого файла завершилась с ошибкой.');
      }
      window.setTimeout(() => document.querySelector('.result, .progress')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 0);
    } catch (e) {
      setError(`Не удалось открыть «${entry.filename}»: ${message(e)}`);
    } finally { locked.current = false; setBusy(null); }
  }
  const visible = filterReportResults(results, report, bucketFilter, reasonFilter);
  const sorted = orderResults(visible, sort, report);
  const showBucket = (bucket: ReportBucket) => {
    setBucketFilter(current => current === bucket ? 'all' : bucket);
    setReasonFilter('all');
    if (bucket !== 'accepted') setSort('risk');
  };
  const goToFirstRisk = () => {
    setBucketFilter('needs_review'); setReasonFilter('all'); setSort('risk');
    window.setTimeout(() => document.querySelector('tr[data-risk="true"]')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 0);
  };
  const closeReport = () => {
    storage.removeItem(ACTIVE_JOB_KEY);
    setResults([]); setReport(null); setJobId(null); setFileInfo(null); setFilename(''); setProgress(null);
    setBucketFilter('all'); setReasonFilter('all'); setNotice('Отчёт закрыт. Серверный срок хранения не изменился.');
  };

  return <>
    <header><div className="brand"><span className="brand-icon"><img src={`${import.meta.env.BASE_URL}russia-flag.svg`} width="32" height="32" alt="Флаг России"/></span>Классификатор товаров</div></header>
    <main>
      <section className="intro"><h1>Найдите код товара</h1><p>Код товара и условия закупки по перечням ПП № 1875</p></section>
      <div className="forms">
        <form className="card single" onSubmit={e => { e.preventDefault(); void single(); }}>
          <h2>Один товар</h2><label htmlFor="product">Название товара</label>
          <div className="input-wrap"><input id="product" value={name} maxLength={PRODUCT_NAME_MAX_LENGTH} disabled={!!busy} placeholder="Ноутбук Lenovo IdeaPad 15.6 дюймов"
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
            <UploadIcon/><strong>{busy === 'file' ? 'Загружаем файл…' : filename || 'Перетащите файл сюда'}</strong>
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
        {result && <section className="card result"><ResultCard result={result} onSaved={saveVerdict} onResolve={resolveClarification} onResolved={clarificationResolved}/></section>}
        {results.length > 0 && <section className="card result">
          <div className="result-heading"><h2>Результаты списка</h2>
            <button type="button" className="outline small" onClick={closeReport}>Закрыть отчёт</button>
          </div>
          {report && <div className="report-summary" aria-label="Итоги закупки">
            {(['accepted', 'needs_review', 'not_found'] as ReportBucket[]).map(bucket => {
              const count = bucket === 'accepted' ? report.summary.accepted
                : bucket === 'needs_review' ? report.summary.needsReview : report.summary.notFound;
              return <button type="button" className={`summary-card summary-${bucket}`} key={bucket}
                aria-pressed={bucketFilter === bucket} onClick={() => showBucket(bucket)}>
                <strong>{count}</strong><span>{BUCKET_LABEL[bucket]}</span>
                <small>{report.summary.total ? Math.round(count * 100 / report.summary.total) : 0}%</small>
              </button>;
            })}
            <p className="report-provenance">Автоматически: {report.summary.autoAccepted} · Специалистом: {report.summary.specialistResolved}</p>
          </div>}
          {report && report.summary.needsReview + report.summary.notFound > 0 &&
            <button type="button" className="outline risk-next" onClick={goToFirstRisk}>Перейти к самой рискованной позиции</button>}
          <div className="toolbar"><label htmlFor="sort">Сортировка
            <select id="sort" value={sort} onChange={e => setSort(e.target.value as SortMode)}>
              <option value="original">Как в файле</option><option value="name">По названию</option><option value="confidence">По уверенности</option><option value="risk">Сначала рискованные</option>
            </select></label>
            {report && <label htmlFor="reason-filter">Причина
              <select id="reason-filter" value={reasonFilter} onChange={event => setReasonFilter(event.target.value as ReportReasonCode | 'all')}>
                <option value="all">Все причины</option>
                {Object.entries(report.summary.reasonCounts).map(([code, count]) =>
                  <option value={code} key={code}>{REASON_LABEL[code as ReportReasonCode]} ({count})</option>)}
              </select></label>}
            <span className="toolbar-actions">
              <button type="button" className="outline" disabled={exporting} onClick={async () => { setExporting(true);
                try { await exportResults(sorted, report); } catch (e) { setError(message(e)); } finally { setExporting(false); }
              }}>{exporting ? 'Сохраняем…' : 'Скачать Excel'}</button>
              <button type="button" className="outline" disabled={exporting} onClick={() => {
                try { exportResultsCsv(sorted, report); } catch (e) { setError(message(e)); }
              }}>Скачать CSV</button>
            </span>
          </div>
          <div className="table-scroll"><table><thead><tr><th scope="col" className="row-num">№</th><th scope="col">Товар</th><th scope="col">Код и статус</th><th scope="col">Условия закупки</th><th scope="col">Уверенность</th><th scope="col">Подробности</th></tr></thead>
            <tbody>{sorted.map((r, index) => {
              const sourceIndex = results.indexOf(r);
              const itemReport = reportItem(report, r, sourceIndex);
              return <tr key={sourceIndex} id={`report-item-${r.itemId ?? sourceIndex}`}
                data-risk={itemReport && itemReport.bucket !== 'accepted' ? 'true' : undefined}>
                <td className="row-num">{index + 1}</td>
                <td>{r.name}{itemReport && <span className="report-reasons">{itemReport.reasons.map(reason =>
                  <small key={reason.code}><strong>{REASON_LABEL[reason.code]}:</strong> {reason.message}</small>)}</span>}</td>
                <td>{r.code ?? '—'}{r.code && <span className="cell-status">Категория: {r.title ?? 'название не указано'}</span>}<span className="muted cell-status">{resultStatus(r)}</span></td>
                <td className="procurement-cell"><Procurement result={r} brief/></td>
                <td>{percent(r.confidence)}{r.status === 'not_found' && <small className="muted">Код не подобран</small>}</td><td><SourceLink url={r.url}/>
                  <details className="details"><summary>{r.status === 'needs_clarification' ? 'Уточнить товар' : 'Проверить результат'}</summary><ResultCard result={r} compact onSaved={saveVerdict} onResolve={resolveClarification} onResolved={clarificationResolved}/></details></td>
              </tr>;
            })}</tbody></table></div>
          {sorted.length === 0 && <p className="hint">По выбранному фильтру позиций нет.</p>}
        </section>}
      </div>
      <section className="card file-history" aria-labelledby="file-history-title">
        <div className="history-heading">
          <span className="history-icon" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
              <path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>
            </svg>
          </span>
          <div><h2 id="file-history-title">История файлов</h2>
            <p>Последние три отчёта всегда под рукой</p></div>
          <span className="history-limit">до 3 файлов</span>
        </div>
        {fileHistory.length === 0 ? <div className="history-empty">
          <UploadIcon/><span>Загрузите файл — здесь появится ссылка на его отчёт.</span>
        </div> : <ul className="history-list">{fileHistory.map(entry =>
          <li key={entry.jobId} className={jobId === entry.jobId ? 'history-current' : ''}>
            <button type="button" className="history-open" disabled={!!busy}
              onClick={() => void openHistory(entry)}>
              <span className="history-file-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                  <path d="M14 3H5v18h14V8zM14 3v5h5M8 13h8M8 17h5"/>
                </svg>
              </span>
              <span className="history-file-copy"><strong>{entry.filename}</strong>
                <small>{historyDate(entry.createdAt)} · {entry.total} позиц.</small></span>
              <span className={`history-status history-${entry.status}`}>{FILE_STATUS_LABEL[entry.status]}</span>
              <span className="history-action">{jobId === entry.jobId ? 'Текущий' : 'Открыть'}<span aria-hidden="true">→</span></span>
            </button>
          </li>)}
        </ul>}
        <p className="history-note">В браузере сохраняются только имена файлов и ссылки на отчёты, без названий товаров. Результаты доступны, пока хранится серверное задание.</p>
      </section>
      <footer>{PROCUREMENT_SCOPE}<br/><a href={PROCUREMENT_SOURCE}>Основание: ПП № 1875, пункты 1 и 2</a></footer>
    </main>
  </>;
}
