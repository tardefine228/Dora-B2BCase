import type { ClarificationOption } from './api';

export type CodeLevel = 'group' | 'detail' | 'intermediate';

/** Structural hierarchy only: no claim that one official code is preferable. */
export function clarificationCodeLevel(
  option: ClarificationOption,
  options: readonly ClarificationOption[],
): CodeLevel | null {
  const others = options.filter(candidate => candidate.optionId !== option.optionId);
  const isAncestor = (parent: string, child: string) => {
    if (child.startsWith(`${parent}.`)) return true;
    const last = parent.split('.').at(-1) ?? '';
    if (last.length !== 1 || !child.startsWith(parent)) return false;
    const suffix = child.slice(parent.length);
    return /^\d(\..+)?$/.test(suffix);
  };
  const hasParent = others.some(candidate => isAncestor(candidate.code, option.code));
  const hasChild = others.some(candidate => isAncestor(option.code, candidate.code));
  if (hasParent && hasChild) return 'intermediate';
  if (hasChild) return 'group';
  if (hasParent) return 'detail';
  return null;
}

export type ClarificationOptionGroup = {
  key: string;
  code: string;
  title: string;
  options: ClarificationOption[];
  clauseRefs: string[];
};

/**
 * One visual choice per code and official description. Equal codes with different
 * descriptions stay separate: in the decree they may be unrelated legal rows.
 */
export function groupClarificationOptions(
  options: readonly ClarificationOption[],
): ClarificationOptionGroup[] {
  const groups = new Map<string, ClarificationOptionGroup>();
  for (const option of options) {
    const key = `${option.code}\u0000${option.title.trim().toLocaleLowerCase('ru')}`;
    const refs = (option.clauseRef ?? '').split('; ').map(ref => ref.trim()).filter(Boolean);
    const existing = groups.get(key);
    if (existing) {
      existing.options.push(option);
      existing.clauseRefs = [...new Set([...existing.clauseRefs, ...refs])];
    } else {
      groups.set(key, {
        key, code: option.code, title: option.title,
        options: [option], clauseRefs: [...new Set(refs)],
      });
    }
  }
  return [...groups.values()];
}

export type ClarificationTitleGroup = {
  key: string;
  title: string;
  codeGroups: ClarificationOptionGroup[];
  clauseRefs: string[];
};

/** One product card: code choices are nested instead of repeating the same title. */
export function groupClarificationTitles(
  options: readonly ClarificationOption[],
): ClarificationTitleGroup[] {
  const groups = new Map<string, ClarificationTitleGroup>();
  for (const codeGroup of groupClarificationOptions(options)) {
    const key = codeGroup.title.trim().toLocaleLowerCase('ru');
    const existing = groups.get(key);
    if (existing) {
      existing.codeGroups.push(codeGroup);
      existing.clauseRefs = [...new Set([...existing.clauseRefs, ...codeGroup.clauseRefs])];
    } else {
      groups.set(key, {
        key, title: codeGroup.title, codeGroups: [codeGroup],
        clauseRefs: [...codeGroup.clauseRefs],
      });
    }
  }
  return [...groups.values()];
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
