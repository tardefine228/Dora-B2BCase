# Dora B2BCase

Backend API для классификации товаров по ПП № 1875. Сейчас API использует временную заглушку: настоящий ML-поиск ещё не подключён.

## Запуск

```bash
docker compose up --build
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Qdrant: http://localhost:6333

## Один товар

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"name":"Ноутбук Lenovo IdeaPad"}'
```

## Пачка с прогрессом

Отправить до 100 товаров:

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H 'Content-Type: application/json' \
  -d '{"items":["Ноутбук Lenovo","Мышь компьютерная"]}'
```

API вернёт `job_id` и `status_url`. Фронтенд должен периодически запрашивать `GET` по `status_url`. Ответ содержит статус, число обработанных товаров, процент выполнения и уже готовые результаты.

Состояние задач сейчас хранится в памяти API. При перезапуске сервиса незавершённые задачи пропадут; для текущего прототипа это допустимо.

Уверенность в API задаётся числом от `0` до `1`; интерфейс может показывать её в процентах. Эндпоинты пока возвращают демонстрационные данные.

## Тесты API

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api python -m pytest api/tests -q
```
