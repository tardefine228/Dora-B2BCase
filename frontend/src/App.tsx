import { useEffect, useRef, useState } from 'react';
import { api, type Result, type Status } from './api';
import { readTable, namesFrom, exportResults, type Table } from './files';

function Icon({ kind = 'search' }: { kind?: 'search' | 'upload' }) {
  return <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">{kind === 'search' ? <><circle cx="10" cy="10" r="6.5"/><path d="m15 15 6 6"/></> : <><path d="M14 3H5v18h14V8zM14 3v5h5M12 17V10m-4 4 4-4 4 4"/></>}</svg>;
}

const STATUS_LABEL: Record<Status, string> = {
  ok: 'Уверенный результат',
  review: 'Требует проверки',
  not_found: 'Не найдено',
};

const message = (error: unknown) => error instanceof Error ? error.message : 'Не удалось выполнить действие. Попробуйте ещё раз.';
const percent = (result: Result) => result.confidence === null ? '—' : `${Math.round(result.confidence * 100)}%`;

// Ссылка на пункт постановления: относительная `/source/...` или абсолютная http(s).
const isSafeUrl = (url: string | null): url is string => !!url && (/^\/source\//.test(url) || /^https?:\/\//i.test(url));

function Source({ result }: { result: Result }) {
  if (!isSafeUrl(result.url)) return <span className="muted">Ссылка на пункт недоступна</span>;
  return <a href={result.url} target="_blank" rel="noreferrer">Открыть пункт постановления ↗</a>;
}

export default function App() {
  const [name, setName] = useState('');
  const [busy, setBusy] = useState<'single' | 'file' | 'batch' | null>(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [table, setTable] = useState<Table | null>(null);
  const [filename, setFilename] = useState('');
  const [column, setColumn] = useState(0);
  const [progress, setProgress] = useState<{ processed: number; total: number; state: 'running' | 'done' | 'failed' } | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [sort, setSort] = useState<'original' | 'name' | 'confidence'>('original');
  const [drag, setDrag] = useState(false);
  const [exporting, setExporting] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  const locked = useRef(false);
  useEffect(() => () => { generation.current++; }, []);

  async function single() {
    if (locked.current) return;
    const value = name.trim();
    if (!value) { setError('Введите название товара.'); return; }
    locked.current = true; setBusy('single'); setError(''); setResult(null); setResults([]); setProgress(null);
    try { setResult(await api.classify(value)); } catch (e) { setError(message(e)); }
    finally { locked.current = false; setBusy(null); }
  }

  async function load(file?: File) {
    if (!file || locked.current) return;
    locked.current = true; setBusy('file'); setError(''); setTable(null); setFilename(''); setResults([]); setProgress(null); setResult(null);
    try {
      const value = await readTable(file); setTable(value); setFilename(file.name);
      const preferred = value.headers.findIndex(h => /^(товар|название|наименование|name|product_name)$/i.test(h));
      setColumn(preferred < 0 ? 0 : preferred);
    } catch (e) { setError(message(e)); }
    finally { locked.current = false; setBusy(null); if (input.current) input.current.value = ''; }
  }

  async function batch() {
    if (!table || locked.current) return;
    let names: string[];
    try { names = namesFrom(table, column); } catch (e) { setError(message(e)); return; }
    locked.current = true;
    const current = ++generation.current;
    setBusy('batch'); setError(''); setResults([]); setResult(null); setProgress({ processed: 0, total: names.length, state: 'running' });
    try {
      const { job_id } = await api.startBatch(names);
      const deadline = Date.now() + 180000;
      while (current === generation.current) {
        if (Date.now() > deadline) throw new Error('Ожидание превысило 3 минуты. Попробуйте снова.');
        const job = await api.getBatch(job_id);
        if (current !== generation.current) return;
        setProgress({ processed: job.processed, total: job.total, state: 'running' });
        if (job.status === 'completed') {
          setResults(job.results);
          setProgress({ processed: job.processed, total: job.total, state: 'done' });
          break;
        }
        if (job.status === 'failed') {
          setResults(job.results);
          setProgress({ processed: job.processed, total: job.total, state: 'failed' });
          throw new Error(job.error ?? 'Обработка списка завершилась ошибкой.');
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (e) { if (current === generation.current) setError(message(e)); }
    finally { if (current === generation.current) { locked.current = false; setBusy(null); } }
  }

  const sorted = [...results].sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name, 'ru') : sort === 'confidence' ? (b.confidence ?? -1) - (a.confidence ?? -1) : 0);

  return <>
    <header><div className="brand"><span className="brand-icon"><Icon/></span>Классификатор товаров</div></header>
    <main>
      <section className="intro"><h1>Найдите код товара</h1><p>Поиск позиции в перечне ПП № 1875 по названию товара</p></section>
      <div className="forms">
        <form className="card single" onSubmit={e => { e.preventDefault(); void single(); }}>
          <h2>Один товар</h2><label htmlFor="product">Название товара</label>
          <div className="input-wrap"><input id="product" value={name} maxLength={500} disabled={!!busy} placeholder="Ноутбук Lenovo IdeaPad 15.6 дюймов" onChange={e => { setName(e.target.value); setResult(null); }}/><button type="button" className="clear" aria-label="Очистить название" disabled={!!busy || !name} onClick={() => { setName(''); setResult(null); }}>×</button></div>
          <p className="hint">Укажите тип товара, модель и основные характеристики</p>
          <button className="primary submit" disabled={!!busy}>{busy === 'single' ? 'Определяем код…' : 'Определить код'} <span aria-hidden="true">→</span></button>
        </form>
        <section className="card"><h2>Список товаров</h2>
          <div className={`drop ${drag ? 'drag' : ''}`} onDragOver={e => { e.preventDefault(); if (!busy) setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length > 1) { setError('Загрузите один файл за раз.'); return; } void load(e.dataTransfer.files[0]); }}>
            <Icon kind="upload"/><strong>{busy === 'file' ? 'Читаем файл…' : filename || 'Перетащите файл сюда'}</strong><span>Excel (.xlsx) или CSV · до 100 товаров</span>
          </div>
          <input ref={input} type="file" accept=".xlsx,.csv" hidden onChange={e => void load(e.target.files?.[0])}/>
          <button className="outline wide" disabled={!!busy} onClick={() => input.current?.click()}>{filename ? 'Выбрать другой файл' : 'Выбрать файл'}</button>
          {table && <div className="file-options"><label htmlFor="column">Столбец с названиями</label><select id="column" value={column} disabled={!!busy} onChange={e => { setColumn(Number(e.target.value)); setResults([]); setProgress(null); }}>{table.headers.map((h, i) => <option key={i} value={i}>{h}</option>)}</select><p className="hint">Строк товаров: {table.rows.length}. Первый лист, первая строка — заголовки.</p><button className="primary wide" disabled={!!busy} onClick={() => void batch()}>Обработать список</button><button className="text-button wide" disabled={!!busy} onClick={() => { setTable(null); setFilename(''); setResults([]); setProgress(null); }}>Удалить файл</button></div>}
        </section>
      </div>
      {error && <div className="error" role="alert">{error}</div>}
      <div aria-live="polite" aria-busy={!!busy}>
      {progress && <section className="card progress"><strong>{progress.state === 'running' ? 'Обрабатываем список…' : progress.state === 'done' ? 'Обработка завершена' : 'Обработка остановлена'}</strong><span>{progress.processed} из {progress.total}</span><progress max={progress.total} value={progress.processed}/></section>}
      {result && <section className="card result"><div className="result-heading"><h2>Результат</h2><span className="badge">{STATUS_LABEL[result.status]}</span></div><div className="result-grid"><div><p className="muted">Предполагаемый код ОКПД2</p>{result.status === 'not_found' ? <p className="title">Надёжного совпадения в перечне не найдено</p> : <><div className="code">{result.code ?? '—'}</div><p className="title">{result.title}</p>{result.clauseRef && <p className="muted">{result.clauseRef}</p>}</>}</div><div className="confidence"><strong>{percent(result)}</strong><p>Уверенность модели</p>{result.confidence !== null && <progress max={1} value={result.confidence}/>}</div></div><div className="source"><Source result={result}/></div></section>}
      {results.length > 0 && <section className="card result"><div className="result-heading"><h2>Результаты списка</h2></div><div className="toolbar"><label>Сортировка <select value={sort} onChange={e => setSort(e.target.value as typeof sort)}><option value="original">Как в файле</option><option value="name">По названию</option><option value="confidence">По уверенности</option></select></label><button className="outline" disabled={exporting} onClick={async () => { setExporting(true); try { await exportResults(sorted); } catch (e) { setError(message(e)); } finally { setExporting(false); } }}>{exporting ? 'Сохраняем…' : 'Скачать Excel'}</button></div><div className="table-scroll"><table><thead><tr><th>Товар</th><th>Код ОКПД2</th><th>Уверенность</th><th>Результат</th></tr></thead><tbody>{sorted.map((r, i) => <tr key={i}><td>{r.name}</td><td className="nowrap">{r.code ?? '—'}</td><td>{percent(r)}<div className="muted">{STATUS_LABEL[r.status]}</div></td><td>{r.status === 'not_found' ? 'Надёжного совпадения в перечне не найдено' : r.title}{r.clauseRef && <div className="muted">{r.clauseRef}</div>}<div><Source result={r}/></div></td></tr>)}</tbody></table></div></section>}
      </div>
      <footer>Подбор кода — подсказка, а не юридический вердикт.<br/>Результат требует проверки специалистом.</footer>
    </main>
  </>;
}
