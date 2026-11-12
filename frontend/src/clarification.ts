import type { ClarificationOption } from './api';

export type CodeLevel = 'group' | 'detail' | 'intermediate';

/** Structural hierarchy only: no claim that one official code is preferable. */
export function clarificationCodeLevel(
  option: ClarificationOption,
  options: readonly ClarificationOption[],
): CodeLevel | null {
  const others = options.filter(candidate => candidate.optionId !== option.optionId);
  const hasParent = others.some(candidate => option.code.startsWith(`${candidate.code}.`));
  const hasChild = others.some(candidate => candidate.code.startsWith(`${option.code}.`));
  if (hasParent && hasChild) return 'intermediate';
  if (hasChild) return 'group';
  if (hasParent) return 'detail';
  return null;
}

export const CODE_LEVEL_LABEL: Record<CodeLevel, string> = {
  group: 'Укрупнённый код',
  detail: 'Детальный код',
  intermediate: 'Промежуточный уровень кода',
};

/** Human-readable audit value shared by CSV and XLSX exporters. */
export function clarificationOptionExport(
  option: ClarificationOption,
  options: readonly ClarificationOption[],
  includeId = true,
): string {
  const level = clarificationCodeLevel(option, options);
  const evidence = [
    option.clauseRef,
    level ? CODE_LEVEL_LABEL[level] : null,
  ].filter(Boolean).join('; ');
  const id = includeId ? `${option.optionId}: ` : '';
  const suffix = evidence ? ` [${evidence}]` : '';
  return `${id}${option.code} — ${option.title}${suffix}`;
}
