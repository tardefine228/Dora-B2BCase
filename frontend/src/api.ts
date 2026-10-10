// Демонстрационный контракт. Согласовать поля с бэкендом перед подключением.
export type Result = { name: string; code: string | null; title: string; confidence: number | null; url: string | null };
export type Job = { job_id: string; status: 'processing' | 'completed'; processed: number; total: number; results: Result[] };
export interface ClassifierApi {
  classify(name: string): Promise<Result>;
  startBatch(names: string[]): Promise<{ job_id: string }>;
  getBatch(id: string): Promise<Job>;
}
const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));
const jobs = new Map<string, { names: string[]; started: number }>();
function example(name: string): Result {
  // Фикстура для проверки интерфейса, не поиск по перечню и не ML-модель.
  if (/ноутбук|laptop/i.test(name)) return { name, code: '26.20.11.110', title: 'Компьютеры портативные', confidence: 0.94, url: null };
  return { name, code: null, title: 'Нет демонстрационного ответа для этого товара', confidence: null, url: null };
}
export const api: ClassifierApi = {
  async classify(name) { await delay(900); return example(name); },
  async startBatch(names) {
    const job_id = crypto.randomUUID();
    jobs.set(job_id, { names: [...names], started: Date.now() });
    return { job_id };
  },
  async getBatch(id) {
    await delay(180);
    const job = jobs.get(id);
    if (!job) throw new Error('Задание не найдено. Запустите обработку ещё раз.');
    const processed = Math.min(job.names.length, Math.floor((Date.now() - job.started) / 150));
    const done = processed === job.names.length;
    const value: Job = { job_id: id, status: done ? 'completed' : 'processing', processed, total: job.names.length, results: done ? job.names.map(example) : [] };
    if (done) jobs.delete(id);
    return value;
  },
};
// Будущее подключение в этом модуле:
// startBatch -> POST /classify/batch; getBatch -> GET /classify/batch/{job_id}.
// Формат тела запроса и ответа пока неизвестен. Здесь нет выдуманного HTTP-контракта.
