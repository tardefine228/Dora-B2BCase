# Запуск исправлений поиска в Docker: Windows и PowerShell

Этот архив содержит изменения для существующего Dora-B2BCase, а не весь проект.
Основа: main, commit c18f2c4f626586478889f2d7f7f1dd976e7333e5, проверенный 23.10.2026.
Ссылку на постановление, данные каталога и интерфейс эти изменения не меняют.
Прирост до 85% требуется подтвердить измерением: сам архив этого не гарантирует.

## 1. Подготовить проект

Откройте Docker Desktop и дождитесь запуска Docker Engine. Нужен режим Linux containers.
Откройте PowerShell в существующей папке проекта, где находятся docker-compose.yml, api, ml, frontend.
Можно открыть эту папку в VS Code и выбрать «Терминал → Создать терминал».

```powershell
docker version
docker compose version
Get-Location
Test-Path .\docker-compose.yml
git status --short
```

Последний Test-Path должен вывести True. Если Docker отвечает ошибкой подключения к daemon,
сначала дождитесь его запуска. Инструкция установки при необходимости:
https://docs.docker.com/desktop/setup/install/windows-install/

Сохраните текущие локальные правки. Если проект уже работает, используйте ту же папку, тот же .env
и прежнее имя Compose-проекта: другое имя создаст отдельные volumes и будет выглядеть как пустая база.
Не выполняйте git reset --hard или git pull поверх несохранённых правок для применения этого пакета.

## 2. Скопировать исправления

Распакуйте архив в отдельную папку. Скопируйте в корень вашего проекта папки api, eval, ml, scripts
из архива, объединяя их с существующими и подтверждая замену только совпадающих файлов.
Не удаляйте существующие папки проекта перед копированием.

Заменяются:

- ml/embedder.py
- ml/search.py
- scripts/download_model.py

Добавляются:

- eval/acceptance_eval.py
- api/tests/test_embedding_inputs.py
- api/tests/test_acceptance_eval.py
- docker-compose.search.yml — скопируйте этот файл отдельно в корень проекта.

Файл search_accuracy.patch содержит те же изменения Python для просмотра diff.
После копирования файлов применять patch повторно не нужно.

Если ваш код существенно изменился после указанного commit, сначала сравните три заменяемых файла.
Основной docker-compose.yml и существующий .env сохраняются. Для совершенно нового запуска без .env
можно использовать значения основного Compose по умолчанию; этот пакет не требует перезаписывать .env.

Важный известный блокер: в проверенной main оставались маркеры конфликта Git в frontend.
Посмотреть, есть ли они в вашей копии:

```powershell
Select-String -Path .\frontend\src\App.tsx,.\frontend\src\styles.css -Pattern '^(<<<<<<<|=======|>>>>>>>)'
```

Если команда нашла строки, сначала примените ранее подготовленное исправление фронтенда
(App.tsx, files.ts, styles.css). Этот архив содержит только изменения поиска, а общий Dockerfile
собирает и фронтенд. Ошибка npm run build или TypeScript в этой ситуации блокирует всю сборку.

## 3. Измерить результат до пересборки — если старый API уже работает

Копирование файлов на Windows ещё не обновляет уже запущенный контейнер. Это удобный момент
снять исходный результат. Для нового проекта, где контейнеров пока нет, пропустите этот шаг.

```powershell
docker compose cp eval/acceptance_eval.py api:/tmp/acceptance_eval.py
docker compose cp eval/labeled_examples.csv api:/tmp/dora-eval.csv
docker compose exec -T api python /tmp/acceptance_eval.py --dataset /tmp/dora-eval.csv --api-url http://127.0.0.1:8000 --output /tmp/acceptance_before.json
docker compose cp api:/tmp/acceptance_before.json ./acceptance_before.json
```

Команды выполняйте по очереди. Если evaluator выводит FAIL и возвращает код 1, всё равно выполните
последнюю команду: FAIL означает, что точность ниже порога, отчёт при этом сохраняется.
Если не удалось скопировать скрипт или контейнер не запущен, сначала исправьте эту ошибку.

Тест делает реальные запросы к приложению, они могут попасть в обычный журнал запросов.
В отчёт входят все строки, включая ошибки HTTP и таймауты. Поэтому сначала проверьте поле errors.

## 4. Проверить конфигурацию и собрать образ

Дополнительный docker-compose.search.yml явно выбирает гибридный catalog + векторы E5.
Настройки одинаковы для api и worker. Основной Compose продолжает задавать БД, очередь, порты и volumes.
Во всех дальнейших командах используйте оба файла в указанном порядке.

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.search.yml build api
```

Если первая команда завершилась с ошибкой, не переходите к сборке. Она проверяет корректность конфигурации.
Сборка требует интернет для Python-зависимостей, PyTorch, Hugging Face и весов E5 и может идти долго.
Закрывать терминал во время сборки не нужно. Предварительно ставить Python, Node.js или PyTorch
на Windows для этих команд не требуется.

У проекта уже есть fallback в Dockerfile: ошибка загрузки transformer иногда оставляет сборку
успешной. Поэтому сообщения WARNING о пропуске transformer нельзя считать успешной загрузкой E5.
Фактическую модель проверяем на шаге 6.

## 5. Запустить сервисы

После успешной сборки:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml up -d
docker compose -f docker-compose.yml -f docker-compose.search.yml ps
docker compose -f docker-compose.yml -f docker-compose.search.yml logs --tail=100 api worker
```

up -d запускает также postgres, redis и qdrant. api и worker используют один образ dora-app.
При первом старте модель загружается в память и создаётся векторный индекс; готовность не мгновенная.
Для наблюдения за запуском:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml logs -f api worker
```

Ctrl+C завершит просмотр логов, а контейнеры продолжат работу.

Открыть приложение: http://localhost:8000/ui/
Документация API: http://localhost:8000/docs
Если в вашем .env изменён API_PORT, используйте его вместо 8000 в ссылках с Windows.
Внутри контейнера API по-прежнему слушает 8000.

## 6. Убедиться, что работает исправленная E5

Откройте http://localhost:8000/health/ready или выполните в PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health/ready | ConvertTo-Json -Depth 10
```

В сведениях о vector_search проверьте:

- enabled: true;
- embedder содержит st-multilingual-e5-small-v2-;
- dim: 384;
- store: qdrant, если внешний векторный сервис доступен. memory означает локальный fallback;
- scoring: vector-features — если загружен обученный под эту модель reranker.

scoring=lexical-reranker означает, что E5 может искать кандидатов, но ранжирование выполняет
лексический артефакт. Текущий Dockerfile может выбрать этот вариант через --gate, если векторный
reranker оказался хуже на его наборе. Это отдельная конфигурация, её нужно фиксировать в замерах.
enabled=false, tfidf или spacy не подтверждают запуск исправленной E5, даже если страница открылась.

Для прямой проверки префиксов модели внутри контейнера:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml exec -T api python -c "from ml.embedder import get_encoder; e=get_encoder(); print(e.tag); print(repr(e._query_prefix), repr(e._doc_prefix))"
```

Ожидается tag с v2 и строка 'query: ' 'passage: '. Команда дополнительно загружает модель в память
на время проверки; если контейнеру не хватает памяти, сначала проверьте его логи и лимиты.

## 7. Измерить точность после обновления

Скрипт запускается внутри уже готового контейнера и обращается к работающему API.
Обычный Dockerfile не копирует новый evaluator в образ, поэтому явно копируем его в /tmp.

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml cp eval/acceptance_eval.py api:/tmp/acceptance_eval.py
docker compose -f docker-compose.yml -f docker-compose.search.yml cp eval/labeled_examples.csv api:/tmp/dora-eval.csv
docker compose -f docker-compose.yml -f docker-compose.search.yml exec -T api python /tmp/acceptance_eval.py --dataset /tmp/dora-eval.csv --api-url http://127.0.0.1:8000 --output /tmp/acceptance_after.json
docker compose -f docker-compose.yml -f docker-compose.search.yml cp api:/tmp/acceptance_after.json ./acceptance_after.json
```

Как и до обновления, FAIL не мешает копированию отчёта. В PowerShell можно вывести итог:

```powershell
(Get-Content .\acceptance_after.json -Raw | ConvertFrom-Json).summary
```

Если есть исходный отчёт:

```powershell
(Get-Content .\acceptance_before.json -Raw | ConvertFrom-Json).summary
(Get-Content .\acceptance_after.json -Raw | ConvertFrom-Json).summary
```

Смотрите exact_top1_all_pct — строгую точность по всем товарам; correct/total — количество верных
ответов; errors — ошибки запросов. Показатель confidence в карточках не заменяет эту метрику.
Также сравните metadata.dataset_sha256: набор до и после должен быть одинаковым.

Для собственного test_sample_100.csv положите файл в корень проекта, затем:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml cp test_sample_100.csv api:/tmp/test_sample_100.csv
docker compose -f docker-compose.yml -f docker-compose.search.yml exec -T api python /tmp/acceptance_eval.py --dataset /tmp/test_sample_100.csv --api-url http://127.0.0.1:8000 --output /tmp/acceptance_100.json
docker compose -f docker-compose.yml -f docker-compose.search.yml cp api:/tmp/acceptance_100.json ./acceptance_100.json
```

После пересоздания контейнера /tmp очищается: повторите копирование evaluator при необходимости.
Разметка теста должна соответствовать версии каталога и согласованному уровню кода. В архиве нет
специально отобранных 20 лёгких товаров; приёмочные 20 должны быть независимыми от настройки модели.

## 8. Повторный запуск и остановка

После перезагрузки компьютера откройте Docker Desktop и из той же папки выполните:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml up -d
```

Остановить контейнеры с сохранением данных:

```powershell
docker compose -f docker-compose.yml -f docker-compose.search.yml stop
```

Не используйте down -v для этого обновления: флаг -v удаляет именованные volumes с данными.
После изменения Python-кода повторите сборку и запуск из шагов 4–5.

## Если что-то не запускается

| Симптом | Что проверить |
|---|---|
| Docker daemon недоступен | Docker Desktop запущен, выбран режим Linux containers |
| Ошибка npm run build / TypeScript / маркеры конфликта | Исправить ранее найденный конфликт frontend; поиск к этой ошибке не относится |
| No module named sentence_transformers / нет модели | Логи build: загрузка могла быть пропущена; восстановить доступ к PyPI, PyTorch и Hugging Face |
| После восстановления сети снова используется неудачный слой сборки | Выполнить сборку с --no-cache один раз, затем снова up -d; будет повторная загрузка зависимостей |
| api exited (137), killed | Проверить память Docker Desktop и API_MEMORY_LIMIT / WORKER_MEMORY_LIMIT в существующем .env; увеличить при достаточной RAM компьютера |
| Старый tag st-embedder или отсутствуют префиксы | Убедиться, что заменён scripts/download_model.py, пересобран образ и пересозданы api/worker |
| Ошибка подключения к Qdrant | Посмотреть logs qdrant; backend catalog может временно перейти на поиск по векторам в памяти |
| HTTP 429 в отчёте | Запустить evaluator с --interval 1, без параллельной нагрузки |
| API_PORT уже занят | Остановить другой экземпляр либо выбрать свободный API_PORT в .env и повторить up -d |

Переиндексация: при выбранном здесь backend=catalog новый tag автоматически создаёт новый кэш
векторов и коллекцию зеркала. Запускать /admin/reindex для этого сценария не требуется.
Отдельный экспериментальный backend=qdrant имеет другой индекс; инструкция намеренно его не включает.

Существующий volume catalog_data сохраняет ваш текущий каталог при обновлении образа.
Пересборка не заменяет автоматически данные в уже созданном volume. Для сопоставимых замеров
сохраняйте один каталог и не удаляйте volume ради «обновления поиска».

## Проверки и границы пакета

Python-изменения прошли 73 целевых теста в предыдущем аудите. Их код в этом архиве тот же.
Для нового Compose-дополнения проверены YAML и согласованность настроек api/worker.
Docker Engine в среде подготовки архива отсутствует: полный docker build/up здесь не выполнялся.
Команда config --quiet на шаге 4 проверяет итоговое объединение файлов в вашей установленной версии Compose.
Остальные найденные проблемы обучения и разметки описаны в SEARCH_AUDIT_RU.md и не объявляются
исправленными этим пакетом. Полная готовность проекта и 85% точности пока не подтверждены.

Документация Docker о порядке объединения файлов:
https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/
