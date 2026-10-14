"""Escaped, standalone reader for extracted source fragments (not an official copy)."""

from html import escape


def render_source(fragment) -> str:
    title = escape(fragment.clause_ref or "Позиция перечня")
    text = escape(fragment.text)
    name = escape(fragment.name or "")
    # Highlight escaped text only; no untrusted HTML is injected.
    if name:
        text = text.replace(name, f"<mark>{name}</mark>")
    pages = f"{fragment.pages.start}–{fragment.pages.end}"
    warning = "Текст обрезан по лимиту." if fragment.truncated else ""
    edition = escape(fragment.edition or "Редакция не указана")
    uploaded_link = (
        f'<a href="{escape(fragment.source_url, quote=True)}" rel="noreferrer">'
        "Ссылка на источник, указанная оператором ↗</a>"
        if fragment.source_url
        else ""
    )
    return f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — ПП № 1875</title><style>
body{{font:17px/1.65 system-ui;margin:0;background:#f5f7fa;color:#182538}}
main{{max-width:900px;margin:32px auto;padding:32px;background:white;border-radius:16px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}
mark{{background:#fff0a6}} .note{{color:#526177;font-size:14px}}
@media(max-width:600px){{main{{margin:0;padding:20px}}}}
</style><main><a href="/ui/">← К классификатору</a>
<p>Постановление Правительства РФ № 1875</p><h1>{title}</h1>
<h2>{name}</h2><p>ОКПД2: {escape(fragment.okpd2 or "не указан")} · Страницы: {pages}</p>
<p class="note">Распознанный фрагмент загруженного документа. Это не официальная публикация.
Актуальность редакции необходимо проверить по исходному НПА. {warning}</p>
<p class="note">{edition}. Данные о редакции указаны загрузившим каталог оператором.</p>
<p>{uploaded_link}</p>
<p><a href="https://publication.pravo.gov.ru/document/0001202412250018" rel="noreferrer">
Первоначальная официальная публикация ↗</a>
<span class="note">(не подтверждает актуальность загруженной редакции)</span></p>
<pre>{text}</pre></main></html>"""
