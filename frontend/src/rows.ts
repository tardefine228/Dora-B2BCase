import type { ProcurementReport, Result } from './api';
import { reportItem } from './report.ts';

export type SortMode = 'original' | 'name' | 'confidence' | 'risk';

/**
 * Порядок строк таблицы результатов.
 *
 * Возвращает новый массив: «Как в файле» — серверный порядок, «По названию» и
 * «По уверенности» — отсортированный. Равные значения разрешаются исходным
 * порядком, поэтому результат детерминирован, а входной массив не мутируется.
 * Нумерацию строк считает таблица — это просто её счётчик 1…N.
 */
export function orderResults(
  results: Result[], sort: SortMode, report: ProcurementReport | null = null,
): Result[] {
  const compare = sort === 'name'
    ? (a: Result, b: Result) => a.name.localeCompare(b.name, 'ru')
    : sort === 'confidence'
      ? (a: Result, b: Result) => (b.confidence ?? -1) - (a.confidence ?? -1)
      : sort === 'risk'
        ? (a: Result, b: Result) => {
          const aIndex = results.indexOf(a);
          const bIndex = results.indexOf(b);
          return (reportItem(report, b, bIndex)?.riskRank ?? -1)
            - (reportItem(report, a, aIndex)?.riskRank ?? -1);
        }
        : () => 0;
  return results
    .map((result, index) => ({ result, index }))
    .sort((a, b) => compare(a.result, b.result) || a.index - b.index)
    .map(row => row.result);
}
