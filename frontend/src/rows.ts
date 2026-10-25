import type { Result } from './api';

export type SortMode = 'original' | 'name' | 'confidence';
// Товар вместе с его номером в файле. Номер — свойство строки, а не позиции на
// экране, поэтому он переживает любую сортировку.
export type NumberedResult = { result: Result; position: number };

/** Нумерует ответы по порядку файла: 1 — первая извлечённая строка. */
export function numberResults(results: Result[]): NumberedResult[] {
  return results.map((result, index) => ({ result, position: index + 1 }));
}

/**
 * Сортирует строки, не трогая номера.
 *
 * Товар остаётся со своим файловым номером при любом режиме: специалист сверяет
 * ответ с исходной выгрузкой, и «19» должно означать девятнадцатую строку файла.
 * Равные значения разрешаются по номеру, поэтому порядок детерминирован.
 */
export function orderResults(rows: NumberedResult[], sort: SortMode): NumberedResult[] {
  const byFile = (a: NumberedResult, b: NumberedResult) => a.position - b.position;
  const compare = sort === 'name'
    ? (a: NumberedResult, b: NumberedResult) => a.result.name.localeCompare(b.result.name, 'ru')
    : sort === 'confidence'
      ? (a: NumberedResult, b: NumberedResult) => (b.result.confidence ?? -1) - (a.result.confidence ?? -1)
      : () => 0;
  return [...rows].sort((a, b) => compare(a, b) || byFile(a, b));
}
