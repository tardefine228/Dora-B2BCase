# Frontend (демо-UI)

Статический интерфейс без сборки: `index.html` + `app.js` + `styles.css`.
Подключается только к публичному контракту API v1.0 (`docs/api-contract.md`):

| Вызов | Назначение |
| --- | --- |
| `POST /classify` | один товар |
| `POST /classify/batch` | список до 100 товаров, ответ — `job_id` + `status_url` |
| `GET /classify/batch/{job_id}` | опрос статуса, `progress_percent`, готовые `results` |
| `GET /classify/batch/{job_id}/events` | SSE-прогресс (`progress`/`done`), основной канал |
| `GET /classify/batch/{job_id}.csv` | серверный CSV (кнопка «или с сервера») |
| `GET /source/{record_id}` | фрагмент постановления — ссылка «источник» в результате |

## Запуск

UI раздаётся самим API: откройте http://localhost:8000/ui/ (корень `/`
перенаправляет туда же). Для отдельного хоста достаточно указать адрес API в
поле «API» на странице (CORS включён, см. `CORS_ALLOW_ORIGINS`).

## Прогресс батча: SSE с фолбэком на опрос

1. После `POST /classify/batch` UI открывает `EventSource` на
   `{status_url}/events` и слушает `progress` и `done`.
2. Если `EventSource` недоступен, стрим не отдал ни одного события за 5 секунд
   (прокси буферизует SSE) или соединение упало — UI молча переключается на
   прежний опрос `GET {status_url}` каждые 350 мс. Пользователь видит тот же
   прогресс, канал связи с сервером его не касается.

## Экспорт CSV

Кнопка «Скачать CSV» появляется на завершённом батче и собирает файл на клиенте
из JSON статуса (те же 13 колонок, что у серверного `GET /classify/batch/{id}.csv`):

```
query;code;title;status;confidence;clause_ref;clause_url;
alt1_code;alt1_title;alt2_code;alt2_title;alt3_code;alt3_title
```

Разделитель `;`, CRLF, кодировка UTF-8 с BOM (Excel на русской локали), кавычки
в значениях удваиваются, поля с `;`/кавычками/переводами строк экранируются.
Рядом — ссылка «или с сервера» на серверный эндпоинт (полезно для больших
батчей и скриптов). Правила формата закреплены тестами
`api/tests/test_batch_export.py`; серверный и клиентский экспорт сверялись
побайтово вручную (curl + node) на одном и том же батче.

## Согласование формата

Поля ответа, на которые опирается `app.js` (контракт v1.0, изменения только
аддитивные):

```
query, code, title, confidence, status(ok|review|not_found),
clause_ref, clause_url, alternatives[{code,title,confidence,clause_ref,clause_url}],
latency_ms, contract_version
```

Любое изменение этих полей — повод обновить `docs/api-contract.md` и
`api/tests/test_contract.py`, которые держат контракт.
