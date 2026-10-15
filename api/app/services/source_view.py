"""Readable, escaped source view with a highlighted catalog position."""

import re
from html import escape
from urllib.parse import urlsplit

GOVERNMENT_URL = "https://government.ru/docs/all/157130/"
PUBLICATION_URL = "https://publication.pravo.gov.ru/document/0001202412250018"
_POSITION = re.compile(r"^\s*(\d{1,3})\.\s+(?!\d)(.+)$")


def _flow(lines: list[str]) -> str:
    """Join PDF line wraps while keeping real hyphenated words intact."""
    text = "\n".join(lines).replace("\u00ad\n", "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)
    return " ".join(text.split())


def _positions(text: str) -> list[tuple[str, str]]:
    result = []
    number = None
    lines: list[str] = []
    for line in text.splitlines():
        match = _POSITION.match(line)
        if match:
            if number is not None:
                result.append((number, _flow(lines)))
            number, first = match.groups()
            lines = [first]
        elif number is not None:
            stripped = line.strip()
            if not stripped or stripped.isdigit() or stripped.startswith("==="):
                continue
            if re.match(
                r"^(?:Код товара|по Общероссийскому|классификатору|Наименование товара|"
                r"по видам экономической|деятельности ОК |КПЕС 2008)",
                stripped,
            ):
                continue
            lines.append(line)
    if number is not None:
        result.append((number, _flow(lines)))
    return result


def _safe_url(value: str | None) -> str | None:
    try:
        parsed = urlsplit(value or "")
        if parsed.scheme in {"https", "http"} and parsed.netloc:
            return escape(value, quote=True)
    except ValueError:
        pass
    return None


def render_source(fragment) -> str:
    title = escape(fragment.clause_ref or "Позиция перечня")
    name = escape(_flow([fragment.name or ""]))
    code = escape(fragment.okpd2 or "не указан")
    start, end = fragment.pages.start, fragment.pages.end
    pages = str(start) if start == end else f"{start}–{end}"
    page_label = "Страница" if start == end else "Страницы"
    page_info = f"{page_label} {escape(pages)}" if start is not None else ""
    position = re.search(r"позиция\s+(\d+)", fragment.clause_ref or "", re.IGNORECASE)
    blocks = _positions(fragment.text)
    target = position[1] if position else None
    selected = [text for number, text in blocks if number == target]

    if selected:
        excerpt = " ".join(selected)
        # OCR places a table's code halfway through a wrapped name. The code
        # has its own badge above; remove only this exact code from the excerpt.
        if fragment.okpd2:
            excerpt = re.sub(rf"(?<![\d.]){re.escape(fragment.okpd2)}(?![\d.])", "", excerpt)
        highlighted = escape(" ".join(excerpt.split()))
        caption = "Текст выбранной позиции"
    else:
        # Never mark a different row when OCR has lost the selected row number.
        highlighted = name or escape(_flow(fragment.text.splitlines()))
        caption = "Наименование позиции в каталоге"

    context_parts = []
    for number, text in blocks:
        content = f"<b>{escape(number)}.</b> {escape(text)}"
        css = ' class="selected"' if number == target else ""
        context_parts.append(f"<p{css}>{content}</p>")
    context = "".join(context_parts) or f"<p>{escape(_flow(fragment.text.splitlines()))}</p>"
    uploaded = _safe_url(fragment.source_url)
    extra_link = (
        f'<a href="{uploaded}" target="_blank" rel="noopener noreferrer">Источник каталога ↗</a>'
        if uploaded and uploaded not in {GOVERNMENT_URL, PUBLICATION_URL}
        else ""
    )
    truncated = '<p class="muted">Показана часть текста страницы.</p>' if fragment.truncated else ""

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — ПП № 1875</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f3f5fa;color:#19263b;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1000px;margin:36px auto;padding:36px;background:#fff;border-radius:20px;
box-shadow:0 8px 32px #21345509}}
a{{color:#3754b8;text-decoration:none}}a:hover{{text-decoration:underline}}
nav{{margin-bottom:26px}}.eyebrow{{color:#64748b;font-size:14px;margin:0 0 8px}}
h1{{font-size:clamp(24px,4vw,34px);line-height:1.25;margin:0 0 18px}}
.meta{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 26px}}
.badge{{background:#eef2ff;color:#354c8f;border-radius:8px;padding:6px 12px;font-size:14px}}
.caption{{font-size:13px;color:#52637a;margin:0 0 12px}}
.excerpt{{margin:0;font-size:19px;line-height:1.65;overflow-wrap:anywhere}}
mark{{color:inherit;background:#fff2ac;padding:2px 3px;box-decoration-break:clone;
-webkit-box-decoration-break:clone;border-radius:3px}}
.links{{display:flex;flex-wrap:wrap;gap:12px 22px;border-top:1px solid #e6eaf1;
margin-top:28px;padding-top:20px;font-size:14px}}
details{{margin-top:28px;border:1px solid #e6eaf1;border-radius:12px;padding:16px 20px}}
summary{{cursor:pointer;font-weight:600}}.context p{{margin:14px 0;overflow-wrap:anywhere}}
.context .selected{{background:#fff8d9;border-left:3px solid #e5b941;padding:12px}}
.muted{{color:#64748b;font-size:14px}}
@media(max-width:640px){{main{{margin:12px;padding:22px;border-radius:14px}}.excerpt{{font-size:17px}}}}
</style></head><body><main>
<nav><a href="/ui/">← К классификатору</a></nav>
<p class="eyebrow">Постановление Правительства РФ от 23.12.2024 № 1875</p>
<h1>{title}</h1>
<div class="meta"><span class="badge">ОКПД2 {code}</span>
<span class="badge">{page_info}</span></div>
<p class="caption">{caption}</p>
<p class="excerpt"><mark>{highlighted}</mark></p>
<div class="links">
<a href="{GOVERNMENT_URL}" target="_blank" rel="noopener noreferrer">На сайте Правительства РФ ↗</a>
<a href="{PUBLICATION_URL}" target="_blank" rel="noopener noreferrer">
Портал правовой информации ↗</a>
{extra_link}</div>
<details><summary>Другие позиции на этой странице</summary>
<div class="context">{context}</div>{truncated}</details>
</main></body></html>"""
