# Dora B2BCase — запуск на Windows 10/11 через Docker Desktop

> Инструкция подготовлена для **ветки `arena/81300b54-dora-b2bcase`** и исправленного архива. Не смешивайте файлы с `main`.
>
> **Проверка:** тесты Python выполнялись в среде подготовки архива. **Docker Desktop там не установлен**, поэтому команды Docker ниже необходимо пройти на вашем компьютере. Фронтенд использует React 19, TypeScript и Vite; сборка `npm ci` не могла завершиться в среде подготовки без доступа к `registry.npmjs.org`.

## 0. Что нужно на компьютере

- Windows 10 (поддерживаемая Docker Desktop редакция) или Windows 11, 64-бит.
- В BIOS/UEFI включена виртуализация; в Windows работает **WSL 2**.
- **Docker Desktop**, выбран режим Linux containers / WSL 2 backend.
- Сеть для скачивания образов (Docker Hub, npm, PyPI). При `INSTALL_TRANSFORMER=true` нужны дополнительно Hugging Face и PyTorch.
- По возможности свободно **от 15–20 ГБ на диске** и **8 ГБ оперативной памяти**, при тяжёлом ML лучше больше.

Проверьте в **PowerShell**:

```powershell
wsl --status
docker --version
docker compose version
docker info
```

Если WSL ещё не установлен, откройте PowerShell **от имени администратора** и выполните `wsl --install`, затем перезагрузите Windows. Установите Docker Desktop с официального сайта https://www.docker.com/products/docker-desktop/ и дождитесь состояния «Engine running». В Settings → General включите **Use the WSL 2 based engine**. Если `docker info` выдаёт ошибку подключения, сначала запустите Docker Desktop и дождитесь готовности.

## 1. Распаковка и конфигурация

1. Распакуйте ZIP **целиком** в папку без длинного пути, например `C:\Projects\Dora-B2BCase`.
2. Откройте PowerShell в корне проекта — там, где есть `docker-compose.yml` и `.env.example`.
3. Создайте собственный `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

В `.env` на первом запуске рекомендуется установить:

```dotenv
API_PORT=8000
CLASSIFIER_BACKEND=catalog
INSTALL_TRANSFORMER=false
API_MEMORY_LIMIT=2g
```

Не меняйте `MODEL_NAME` на локальный Windows-путь: в Docker путь к модели находится внутри контейнера. В настройках по умолчанию `ML_ENABLED=false`, и реальный каталог работает в режиме `catalog` (это **не** mock). Для установки полной модели позже переключите `INSTALL_TRANSFORMER=true`, убедитесь в доступности сайтов Hugging Face / PyTorch, и повторите сборку. Результаты моделей с разными эмбеддерами могут различаться.

**Важно:** в Docker Compose по умолчанию используются `JOB_STORE=postgres`, `QUEUE_BACKEND=celery`, `DATABASE_URL` к контейнеру `postgres`. Настройки `JOB_STORE=sqlite` из `.env.example` относятся к локальному запуску без Compose и при Docker переопределяются. Поэтому в Docker задания и пользовательские вердикты сохраняются в PostgreSQL.

Для совместной работы и публикации поменяйте демонстрационные пароли: установите сложный `POSTGRES_PASSWORD` и непустой `ADMIN_API_KEY` **до создания volume PostgreSQL**. Не загружайте `.env` в GitHub. При изменении пароля уже инициализированной БД потребуется отдельно обновлять пароль в PostgreSQL, а не просто править `.env`.

## 2. Собрать и запустить все сервисы

В корне проекта:

```powershell
docker compose up --build -d
```

Первая сборка загружает образы и Python/npm-пакеты: это может занять значительное время. Смотрите состояние:

```powershell
docker compose ps
docker compose logs --tail 80 api
docker compose logs --tail 80 worker
```

Вы должны увидеть сервисы `api`, `worker`, `postgres`, `redis`, `qdrant` в состоянии работы; `api` должен пройти healthcheck. Если не успел подняться, подождите ещё 1–2 минуты и проверьте логи.

Откройте в браузере:

- **Сайт:** http://localhost:8000/ui/
- **Описание API:** http://localhost:8000/docs
- **Готовность:** http://localhost:8000/health/ready

Проверка через PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/health/ready
```

Проверка классификации:

```powershell
$body = @{ name = 'Электрочайники' } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/classify -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Ожидайте JSON с полями `query`, `code`, `title`, `confidence`, `status`, `clause_url`, `alternatives`. Точный процент **не фиксирован** и зависит от модели.

## 3. Проверить сценарии сайта

1. Введите «Электрочайники», нажмите **Определить код**. Должна появиться карточка с кодом, уверенностью и ссылкой; если модель вернула альтернативы, раскройте их.
2. Нажмите **Код верный**. Сообщение «оценка сохранена» должно появиться только после успешного ответа сервера. Вердикт хранится в PostgreSQL.
3. Загрузите `frontend/examples/products.csv` или собственный `.xlsx`. Сайт отправляет **сам файл** серверу на `/classify/file`; сервер автоматически выбирает столбец, поэтому не требуется ставить npm/ExcelJS для чтения.
4. Дождитесь прогресса и таблицы. Перезагрузите страницу в процессе обработки — незавершённое задание должно восстановиться по `job_id`. Прогресс работает через SSE с запасным polling.
5. Попробуйте **Скачать Excel** и **Скачать CSV**. На компьютере с доступным браузером скачивание должно работать.
6. Сохранить результат можно кнопками **Скачать Excel** / **Скачать CSV**. Локальной истории проверок в интерфейсе больше нет (она хранила названия товаров в браузере и для работы не требовалась).

## 4. Как обновлять код и останавливать

После изменения React-кода или API и повторной упаковки:

```powershell
docker compose up --build -d
```

Посмотреть последние логи:

```powershell
docker compose logs -f api worker
```

Остановить, **сохранив данные**:

```powershell
docker compose down
```

Запустить снова без пересборки:

```powershell
docker compose up -d
```

**Не используйте `docker compose down -v`** без бэкапа: опция `-v` удаляет тома PostgreSQL, каталога и Qdrant вместе с результатами, сохранёнными оценками и другими данными.

> **Запускайте с `-d` (в фоне).** В прикреплённом режиме (`docker compose up` без `-d`) Compose v2
> перехватывает клавиши и **`Ctrl+Z` ставит контейнеры на паузу**, а не сворачивает вывод; после
> этого следующий `docker compose up` падает с `cannot start a paused container`. Остановка — это
> `Ctrl+C`, а если пауза всё-таки случилась — раздел «Последствия паузы» ниже.

## 5. Типичные проблемы

| Симптом | Что проверить |
|---|---|
| `docker` не найден | Установлен ли Docker Desktop, запущен ли Engine, перезапущен ли терминал |
| `Cannot connect to Docker daemon` | Docker Desktop не запущен, WSL 2 выключен или не готов |
| Долго строится образ | Сначала `INSTALL_TRANSFORMER=false`; для полного режима нужны дополнительные большие загрузки |
| `npm ci` / `pip install` не скачивает пакеты | Проверить доступ к registry.npmjs.org, pypi.org и proxy/DNS |
| Порт 8000 занят | Измените `API_PORT=8001` в `.env`, затем откройте `http://localhost:8001/ui/` |
| Сайт открывается, но есть 503 | `docker compose logs --tail 100 api worker`; проверить `/health/ready`, модель и каталог |
| Файл не загрузился | До 9,5 МБ для фронтенда, серверный лимит `MAX_UPLOAD_BYTES`; допустимые типы: .xlsx/.xlsm/.csv/.tsv/.pdf/.docx/.json/.txt/.md |
| Пустые или неверные результаты | Проверьте `/health/ready`, используемый эмбеддер, и настройки `CLASSIFIER_BACKEND`; модель — подсказка, не юридическое заключение |
| История не видна на другом компьютере | История проверок локальная и включается вручную; оценки «верно/неверно» сохраняются отдельно в PostgreSQL |
| `POST /feedback` возвращает 503 | Проверьте доступ к PostgreSQL и `docker compose logs --tail 100 api` |
| `cannot start a paused container, try unpause instead` | Контейнер стоит на паузе (`Ctrl+Z` в прикреплённом `up`, пауза из Docker Desktop или сон/гибернация Windows с WSL 2). Лечится разделом «Последствия паузы» ниже |
| `docker compose ps` показывает `(Paused)` | То же самое: `docker compose unpause`, затем `docker compose up -d` |

### Последствия паузы: `cannot start a paused container`

Пауза — это состояние контейнера, а не ошибка образа: Docker отказывается запускать или
пересоздавать контейнер, который уже существует на паузе, поэтому `docker compose up` падает
целиком, и остальные сервисы (`api`, `worker`) остаются в состоянии `Recreate`. Данные в томах
при этом целы. Порядок восстановления — в корне проекта, где лежит `docker-compose.yml`:

```powershell
cd C:\Projects\Dora-B2BCase          # ваш путь к корню проекта

docker compose ps                      # у кого в STATUS стоит (Paused)
docker compose unpause                 # снять паузу со всех сервисов
docker compose up -d                   # дособрать/пересоздать то, что не поднялось
docker compose ps                      # все сервисы должны быть running, api — healthy
```

Если `unpause` ругается или состояние странное (например, контейнер завис после сна ноутбука),
пересоздайте стек **без удаления томов**:

```powershell
docker compose down                    # без -v: тома и данные сохраняются
docker compose up -d                   # образ dora-app уже собран, --build не нужен
```

Проверка, что всё поднялось:

```powershell
Invoke-RestMethod http://localhost:8000/health/ready
```

Пересборка образа (`--build`) к этой проблеме отношения не имеет: она нужна только после
изменения кода, а не после паузы. Отдельный контейнер можно вернуть и точечно — имя сервиса
видно в `docker compose ps`: `docker unpause dora-b2bcase-qdrant-1`.

## 6. Отдельная разработка фронтенда (без пересборки Docker на каждую кнопку)

Нужно установить **Node.js 22+**. Поднимите бэкенд через Docker, затем откройте **второе** окно PowerShell:

```powershell
cd C:\Projects\Dora-B2BCase\frontend
npm ci
npm run dev
```

Открывайте URL с `/ui/`, который покажет Vite. Прокси Vite передаёт `/classify`, `/feedback` и `/source` в API на `127.0.0.1:8000`. Это режим разработки; для демонстрации команды используйте собранный Docker-сайт на `http://localhost:8000/ui/`.

Всё, что нужно для работы фронтенда, находится в `frontend/src/`: `App.tsx` — страница; `api.ts` — REST и SSE; `files.ts` — экспорт; `styles.css` — оформление.
