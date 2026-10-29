import type { Result } from './api';

export const PROCUREMENT_SCOPE = 'Для закупок по 44-ФЗ и 223-ФЗ. Учитывайте происхождение товара и исключения.';
export const PROCUREMENT_SOURCE = 'https://www.consultant.ru/document/cons_doc_LAW_494318/92d969e26a4326c5d02fa79b8f9cf4994ee5633b/';

export type ProcurementVerdict = {
  title: string;
  explanation: string;
  caution: string | null;
  tone: 'ban' | 'condition' | 'neutral';
};
const rules: Record<number, { title: string; explanation: string }> = {
  1: {
    title: 'Запрет закупки иностранного товара',
    explanation: 'Иностранный товар по общему правилу закупать нельзя. Есть исключения, включая правила ЕАЭС.',
  },
  2: {
    title: 'Ограничение закупки иностранного товара',
    explanation: 'Допуск иностранного товара зависит от условий закупки и наличия подходящей российской заявки с учётом правил ЕАЭС.',
  },
  3: {
    title: 'Минимальная доля российских товаров',
    explanation: 'По 223-ФЗ нужно соблюдать годовую долю российских товаров. Это само по себе не запрет импорта.',
  },
};

/** Explain only evidence returned by the API, never infer permission from no match. */
export function procurementVerdict(result: Pick<Result, 'status' | 'code' | 'clauseRef' | 'codeRole' | 'catalogReviewRequired'>): ProcurementVerdict {
  const unknown = (explanation: string): ProcurementVerdict => ({
    title: 'Условия закупки требуют проверки', explanation, caution: null, tone: 'neutral',
  });
  if (result.status === 'not_found' || !result.code) {
    return unknown('Совпадение не найдено. Это не подтверждает разрешение на закупку иностранного товара.');
  }
  if (result.codeRole === 'Исключение') {
    return unknown('Код отмечен как исключение из позиции. Сверьте текст исключения и другие перечни.');
  }
  if (result.codeRole && result.codeRole !== 'Включён') {
    return unknown('Роль кода не подтверждена. Условия закупки нужно проверить по постановлению.');
  }
  // A result can cite several appendices: do not lose the quota alongside a restriction.
  const refs = (result.clauseRef ?? '').split(';').map(ref => ref.trim());
  const matches = refs.map(ref => /^Приложение\s+(?:№\s*)?(\d+),\s*позиция\s+\d+$/iu.exec(ref));
  if (matches.some(match => !match)) return unknown('Приложение не определено. Проверьте условия по ссылке на постановление.');
  const appendices = [...new Set(matches.map(match => Number(match![1])))];
  if (appendices.some(appendix => !rules[appendix])) {
    return unknown('Для позиции предусмотрены специальные условия. Проверьте их в постановлении.');
  }
  const preliminary = result.status !== 'ok' || result.codeRole !== 'Включён' || result.catalogReviewRequired !== false;
  return {
    title: appendices.map(appendix => rules[appendix].title).join('; '),
    explanation: appendices.map(appendix => rules[appendix].explanation).join(' '),
    caution: preliminary ? 'Предварительный результат — проверьте позицию' : null,
    tone: preliminary ? 'neutral' : appendices.includes(1) ? 'ban' : 'condition',
  };
}
