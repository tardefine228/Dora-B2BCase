# Frontend (демо-UI)

Статический интерфейс без сборки: `index.html` + `app.js` + `styles.css`.
Подключается только к публичному контракту API v1.0 (`docs/api-contract.md`):

| Вызов | Назначение |
| --- | --- |
| `POST /classify` | один товар |
| `POST /classify/batch` | список до 100 товаров, ответ — `job_id` + `status_url` |
| `GET /classify/batch/{job_id}` | опрос статуса, `progress_percent`, готовые `results` |

## Запуск

UI раздаётся самим API: откройте http://localhost:8000/ui/ (корень `/`
перенаправляет туда же). Для отдельного хоста достаточно указать адрес API в
поле «API» на странице (CORS включён, см. `CORS_ALLOW_ORIGINS`).

## Согласование формата

Поля ответа, на которые опирается `app.js` (контракт v1.0, изменения только
аддитивные):

```
query, code, title, confidence, status(ok|review|not_found),
clause_ref, clause_url, alternatives[{code,title,confidence,clause_ref}],
latency_ms, contract_version
```

Любое изменение этих полей — повод обновить `docs/api-contract.md` и
`api/tests/test_contract.py`, которые держат контракт.
