# BGE-M3 в текущей версии проекта: анализ готовности

**Дата:** 2026-10-23 · **Ветка:** `new-irina-branch` (`985b955`) · **Вопрос:** можно ли уже подключать BGE-M3 вместо `intfloat/multilingual-e5-small`.

> **Короткий ответ.** Архитектурно — да, пайплайн модель-агностичен (проверено на синтетическом
> 1024-мерном энкодере: индекс, кэш, Qdrant, поиск работают без правок). Операционно — **нет**:
> `scripts/download_model.py` выбрасывает единственный файл весов BGE-M3 (`pytorch_model.bin`, у модели
> нет `safetensors`), поэтому образ соберётся «с моделью», но рантайм **молча откатится на spaCy**.
> Плюс жёсткий лимит памяти (2 ГБ на контейнер) и главное: в отгруженном реранкере веса
> `dense_cosine`/`dense_gap` равны **0.0**, тегированного артефакта нет — смена эмбеддера сегодня
> не меняет решения. Порядок работ и точные правки — в §4.

---

## 1. Что уже готово (без изменений кода)

| Механизм | Где | Статус для BGE-M3 |
| --- | --- | --- |
| Выбор модели при сборке образа | `docker-compose.yml` → build arg `EMBEDDER_MODEL`, `api/Dockerfile` `ARG EMBEDDER_MODEL` | ✅ `EMBEDDER_MODEL=BAAI/bge-m3 docker compose build` — штатный путь |
| Префиксы запрос/документ | `ml/embedder.py::_e5_prefixes` → для не-e5 пустые строки | ✅ BGE-M3 префиксов не требует; это уже закреплено тестом `api/tests/test_embedding_inputs.py:22` |
| Идентичность модели при копировании в образ | `scripts/download_model.py` пишет `dora_model.json`; `_transformer_spec` строит тег из `model_id`, а не из пути | ✅ тег `st-bge-m3-v2-b118c3d8643d` (у e5 — `st-multilingual-e5-small-v2-9b1a13eefd5f`) |
| Размерность вектора | `ml/indexer.py:124` — берёт фактическую размерность модели, `QDRANT_VECTOR_DIM` (1024) только подсказка | ✅ 1024 = дефолт подсказки, коллекция создаётся корректно |
| Пространства имён кэшей | `ml/dense.py::_build_local` — файл `dense-<tag>-<hash>.npy`; `ml/reranker.py::reranker_artifact_name` — `reranker.<tag>.json` | ✅ векторы и веса одного эмбеддера не переиспользуются другим |
| Деградация «не молча» | `ml/embedder.py::resolve_backend`, `get_encoder` (`explicit` отключает fallback) | ⚠️ работает только при `EMBEDDER_BACKEND=sentence-transformers`; в `auto` — тихий откат |

**Проверка «на живом»:** собрал стенд с энкодером-заглушкой `dim=1024`, `tag=st-bge-m3-v2-b118c3d8643d`
(полный каталог 1126 позиций): `DenseIndex.build` → `dense-st-bge-m3-v2-b118c3d8643d-*.npy`, повторная
сборка читает кэш; `ml.indexer.index_catalog()` создал коллекцию `size=1024`; `ml.search.search_many()`
вернул хиты. То есть ни одного места, где зашита размерность 384 или префиксы e5, в пути данных нет.

## 2. Блокеры

### Б1. Критично: скрипт скачивания оставляет модель без весов

`scripts/download_model.py` скачивает снапшот с `ignore_patterns`, среди которых `pytorch_model.bin`
(«PyTorch weights only: skip ONNX/OpenVINO/TF duplicates»). Для `intfloat/multilingual-e5-small` это
безопасно — в репозитории есть `model.safetensors`. **У `BAAI/bge-m3` safetensors нет**: по HF API в
репозитории лежат `pytorch_model.bin` (2 271 145 830 Б ≈ 2.27 ГБ), `sentencepiece.bpe.model`,
`tokenizer.json` (17 МБ) и каталог `onnx/`.

Прогон того же фильтра на фактическом списке файлов BGE-M3:

```
BGE-M3 -> осталось файлов: 14
есть ли веса: ['colbert_linear.pt', 'sparse_linear.pt']     # это головы sparse/multi-vector, не би-энкодер
выкинуто: ['onnx/model.onnx', 'onnx/model.onnx_data', 'onnx/tokenizer.json', 'pytorch_model.bin']

E5-small -> есть ли веса: ['model.safetensors']              # не затронуто
```

Последствия в Docker-сборке: `download_model.py` завершается с кодом 0 (скачал 14 файлов), `cp -a`
копирует каталог в `/opt/models/embedder`, шаг `build_model.py --dense --gate` падает и глушится
`|| echo "WARNING..."`, образ собирается. В рантайме `MODEL_NAME=/opt/models/embedder` существует как
каталог → `resolve_backend()` выбирает `sentence-transformers` → `SentenceTransformer(...)` бросает
исключение «нет весов» → `get_encoder()` в режиме `auto` уходит на `spacy-ru_core_news_md`.
`/health/ready` при этом покажет `embedder: spacy-ru_core_news_md`, то есть ровно тот сценарий
«сборка успешна, модели нет», который README описывает как самую частую проблему.

### Б2. Жёстко: память

BGE-M3 — 568M параметров, fp32 ≈ 2.27 ГБ только весов; RSS процесса с torch ≈ 3–4 ГБ.
Действующие лимиты:

| Место | Значение | Хватит? |
| --- | --- | --- |
| `.env.example` → `API_MEMORY_LIMIT=1g` | 1 ГБ | ✗ (даже e5-small впритык) |
| `docker-compose.yml` → `${API_MEMORY_LIMIT:-2g}` | 2 ГБ | ✗ |
| `docker-compose.yml` → `${WORKER_MEMORY_LIMIT:-2g}` | 2 ГБ | ✗ (воркер грузит свою копию модели) |
| `QDRANT_MEMORY_LIMIT` | 768 МБ | ✅ (1126 × 1024 × 4 Б ≈ 4.6 МБ векторов) |

Итог: нужны ≥4 ГБ на `api` и на `worker` (или GPU/квантование — но `USE_ONNX`/`USE_INT8` из
`ml/config.py` **нигде не используются**, int8-пути в коде нет).

### Б3. Смысловой блокер: новый эмбеддер не влияет на ответ

- В отгруженном `ml/artifacts/reranker.json` веса `dense_cosine = 0.0`, `dense_gap = 0.0`; файла
  `reranker.<tag>.json` в `ml/artifacts/` нет → `load_reranker(tag)` пишет warning и берёт лексический артефакт.
- Значит эмбеддер участвует только в расширении шорт-листа (`fuse_shortlist`), а не в скоринге.
- Измерено в `docs/audit-2026-10-20.md` (§0 п.4, §3.3): включение/выключение векторного поиска на 274
  размеченных строках меняет **0 кодов и 0 статусов**; потолок поиска почти выбран — правильная позиция
  уже в лексическом top-25 для **97.6 %** запросов с кодом, тогда как dense-top-8 даёт лишь 85.5 %.
- Там же (§0 п.6): dense-признаки дают +1.9 п.п. top-1 на tfidf, но роняют top-1 «по всем», и штатный
  гейт `build_model.py --gate` их удаляет.

Вывод: пока dense-признаки не проходят гейт и не имеют веса, BGE-M3 даст **≈0 изменения качества** —
ровно как e5-small, только медленнее.

### Б4. Цена: образ, сборка, латентность

| Метрика | e5-small (сейчас) | BGE-M3 |
| --- | --- | --- |
| Веса | ~0.47 ГБ (118M, safetensors) | 2.27 ГБ (568M, .bin) |
| Прирост образа | — | +~2.3 ГБ |
| Скачивание при сборке | ~0.5 ГБ | ~2.3 ГБ (один файл >2 ГБ — под прокси/антивирусом рвётся чаще) |
| Латентность CPU на запрос | единицы мс | по замерам аудита семантический режим p50 171 мс против 38.5 мс у каталогового |
| `max_seq_length` | 512 | 8192 по умолчанию; код его **не ограничивает** (в `ml/embedder.py` нет установки `max_seq_length`) |
| `MODEL_BATCH_SIZE` | 16 | 16 рискованно по памяти, нужно 4–8 |

Длина названий в каталоге: медиана 63 символа, p95 — 267, максимум — 1164 (21 название длиннее 512
символов). Для BGE-M3 этого с запасом хватает `max_seq_length=512` (≈1500+ символов русского текста),
8192 контекста не нужны и только увеличивают буферы.

### Б5. Коллекция Qdrant не содержит тег эмбеддера

`ml/indexer.py` пишет в фиксированный `QDRANT_COLLECTION` (`pp1875_items`) через staging + alias.
Смена модели = полная переиндексация и удаление старой коллекции. Для A/B «e5 vs bge-m3 на одном коде»
нужны **разные** `QDRANT_COLLECTION` (например `pp1875_items_bge`), иначе сравнивать нечего.
Замечание: `docs/audit-2026-10-20.md` §7.4 утверждает, что тег уже входит в имя коллекции — в текущей
версии это не так (тег есть только в имени `.npy`-кэша и в имени артефакта реранкера).

### Б6. Мелочь, но легко обжечься

`resolve_backend()` в режиме `auto` включает трансформер только если `Path(MODEL_NAME).is_dir()`.
Если задать `MODEL_NAME=BAAI/bge-m3` (hub ID) без локального каталога — `auto` уйдёт на spaCy, а не
скачает модель. Скачивание в рантайме возможно только при `EMBEDDER_BACKEND=sentence-transformers` и
`MODEL_LOCAL_ONLY=false` (+ доступ к HF из контейнера и писабельный кэш).

## 3. Итоговая таблица готовности

| Вопрос | Ответ |
| --- | --- |
| Нужны ли правки кода, чтобы BGE-M3 *заработал как энкодер*? | Да, одна: не выбрасывать `pytorch_model.bin` (Б1). Плюс проверка «весов нет → падать громко». |
| Нужны ли правки, чтобы он *влиял на решения*? | Да, и они не про модель: `build_model.py --dense` + прохождение гейта, иначе вес dense = 0 (Б3). |
| Хватит ли текущих лимитов compose? | Нет: 2 ГБ против нужных 3.5–4 ГБ на контейнер (Б2). |
| Совместимы ли артефакты (кэш, реранкер, коллекция)? | Кэш и реранкер — да, изолированы тегом. Коллекция — одна на всё, для A/B нужен свой `QDRANT_COLLECTION` (Б5). |
| Есть ли риск «тихо стало хуже/другое»? | Да, главный риск — тихий откат на spaCy (Б1) и тихий пропуск dense-признаков гейтом (Б3). |

## 4. Что сделать, чтобы подключить (по шагам)

**Шаг 0. Починить загрузку весов** (`scripts/download_model.py`) — без этого остальное бессмысленно:

```python
from huggingface_hub import HfApi, snapshot_download

info = HfApi().model_info(repo_id)
files = {s.rfilename for s in info.siblings}
ignore = list(IGNORE_PATTERNS)
if "model.safetensors" not in files:
    ignore.remove("pytorch_model.bin")   # единственный файл весов, например у BAAI/bge-m3
snapshot_download(repo_id=repo_id, local_dir=str(destination), ignore_patterns=ignore)

weights = [p for p in destination.glob("*") if p.suffix in {".bin", ".safetensors"}]
if not weights:
    raise SystemExit(f"В {destination} нет весов модели — проверьте ignore_patterns")
```

Проверка после скачивания обязательна: именно она превращает тихий откат в понятную ошибку сборки.

**Шаг 1. Собрать образ и убедиться, что загрузилась именно BGE-M3:**

```bash
EMBEDDER_MODEL=BAAI/bge-m3 INSTALL_TRANSFORMER=true docker compose build
docker compose exec api python -c "from ml.embedder import get_encoder; e=get_encoder(); print(e.tag, e.dim)"
# ожидается: st-bge-m3-v2-<hash> 1024
curl -s localhost:8000/health/ready | python -m json.tool   # vector_search.embedder / dim
```

**Шаг 2. Поднять лимиты и ужать буферы** (`.env`):

```dotenv
API_MEMORY_LIMIT=4g
WORKER_MEMORY_LIMIT=4g
MODEL_BATCH_SIZE=8
EMBEDDER_BACKEND=sentence-transformers   # запретить тихий откат на spaCy
# MODEL_NAME оставить пустым (compose подставит /opt/models/embedder)
```

и ограничить контекст (`ml/embedder.py`, сразу после загрузки модели):

```python
self._model.max_seq_length = int(os.getenv("MODEL_MAX_SEQ_LENGTH", "512"))
```

**Шаг 3. Переобучить реранкер под новый эмбеддер и проверить, что гейт его оставил:**

```bash
docker compose exec api python scripts/build_model.py --dense --gate --models-dir /app/ml/artifacts
# в логе: "gate: lexical top-1 ... | vector top-1 ..." — артефакт остаётся только при неухудшении
```

**Шаг 4. Снять метрики и свип порогов, сравнить с e5 на одном наборе:**

```bash
PYTHONPATH=api VECTOR_SEARCH=true python eval/run_eval.py --sweep --output eval_report_bge.md
```

Критерии приёмки BGE-M3 (из `docs/audit-2026-10-20.md` §7.7): top-1 ≥ +2 п.п. к e5 при фиксированной
тишине, тишина вне перечня не хуже, p95 `/classify` ≤ 250 мс, RSS ≤ 80 % машины. Если хотя бы один не
выполнен — модель отклоняется, остаётся e5.

**Шаг 5. Откат** — переменными окружения, без пересборки: `EMBEDDER_MODEL=intfloat/multilingual-e5-small`
+ старый артефакт `reranker.json` (лексический) остаётся в образе; для семантического режима —
`SEMANTIC_RERANKER=lexical`.

## 5. Вердикт

- **Подключать сейчас как «улучшение распознавания» — нет.** Даже после починки загрузки модель не
  участвует в решении (вес dense = 0) и упирается в потолок поиска 97.6 %; на имеющемся наборе
  выигрыш будет в пределах шума, а цена — 4 ГБ RAM на контейнер, +2.3 ГБ образа и 3–5× латентности.
- **Подключать как измеряемый A/B — можно, после Б1 + Б2.** Код к этому готов: смена модели — один
  build-arg, артефакты изолированы тегом, откат — переменной окружения. Но перед A/B нужно закрыть Б3:
  добиться, чтобы dense-признаки имели ненулевой вес и проходили гейт **на e5-small** — иначе сравнивать
  будет нечего.
- **Когда ответ станет «да» без оговорок:** dense-признаки участвуют в решении и измерены на e5;
  есть ≥300–500 реальных размеченных запросов от специалистов; лексический recall@25 < 90 % на них
  (сейчас 97.6 %); и есть железо на ≥4 ГБ RAM на контейнер.
