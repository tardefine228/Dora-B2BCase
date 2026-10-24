# Артефакты модели

Здесь лежит всё, что сервис загружает в рантайме. Файлы входят в образ (`api/Dockerfile`),
поэтому их не пересобирают при старте — но и не правят руками: у каждого файла ниже есть
команда-производитель и sha256, а `api/tests/test_shipped_artifacts.py` сверяет их с этим
списком. Если тест упал после пересборки — обновите sha256 в таблице тем же коммитом.

## Что лежит сейчас

| Файл | sha256 | Что это | Чем произведён |
| --- | --- | --- | --- |
| `reranker.json` | `e1f3d8cfc0607b70824f148d3635d1f1ef0f76dc7319088e4b6ae490e3ea6319` | лексический реранкер (веса `dense_*` = 0): работает, когда векторных признаков нет | `python scripts/build_model.py --models-dir models` |
| `reranker.tfidf.json` | `ac0f46fde629b08ddd7037e30914bbb24700514f988168073e52e69d8127caf8` | реранкер с векторными признаками под эмбеддер `tfidf`; **именно он скорит в каталоговом режиме при `VECTOR_SEARCH=true`** | `python scripts/build_model.py --dense --gate --models-dir models` |
| `dense-gate.json` | `3fbe78734b198dc356eb87afc0fdae3d95eed7ca07d6d4da78abbcbd7acdbb7d` | отчёт допуска для `reranker.tfidf.json`: метрики по строкам, sha256 кандидата и базы | тот же прогон `--dense --gate` |
| `tfidf_embedder.json` | `11a73c50d6e903ff56570764cbd392becd9675b651bc926230cffd8f13d03cde` | tfidf-эмбеддер: 512 измерений, 3–4-граммы, 1126 названий | `scripts/build_model.py` (шаг подгонки эмбеддера) |

## Происхождение и воспроизводимость

* Каталог: 984 позиции, `data/data.csv`. Размеченный набор: 274 строки
  (`eval/labeled_examples.csv`), из них 166 с ожидаемым кодом и 108 «вне перечня».
* `reranker.json` **воспроизводится байт в байт**: две независимые полные сборки дают один и тот
  же sha256. Это и проверяет CI-шаг «Shipped artifacts match a rebuild».
* `reranker.tfidf.json` **не воспроизводится побайтово**: два прогона дают 38012 и 38010
  обучающих пар (`dense_cosine` 0.35271 против 0.35267), потому что шортлист кандидатов зависит от
  порядка сложений в BLAS. Метрики на размеченном наборе при этом совпадают построчно, поэтому
  `dense-gate.json` хранит sha256 проверенного файла, а допуск сравнивает метрики, а не хеши.
* `tfidf_embedder.json` различается между сборками только полем `trained_at`; частоты документов
  совпадают — это проверяет
  `test_shipped_tfidf_embedder_is_fitted_on_the_current_catalog`.
* `dense-gate.json` — отчёт последнего прогона допуска. Его `baseline.sha256` равен sha256
  отгруженного `reranker.json`: база переобучается тем же кодом в том же прогоне, поэтому отчёт
  доказуемо относится к текущей паре файлов.

## Измерено на текущих артефактах

Каталоговый режим, эмбеддер `tfidf`, 274 строки размеченного набора
(`EMBEDDER_BACKEND=tfidf eval/run_eval.py`):

| Конфигурация | top-1 (с кодом) | top-3 | верный `not_found` | top-1 по всем | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `VECTOR_SEARCH=false` → `reranker.json` | 82.5 % | 87.3 % | 85.2 % | 83.6 % | 0.053 |
| `VECTOR_SEARCH=true` → `reranker.tfidf.json` | **84.3 %** | **89.8 %** | 85.2 % | **84.7 %** | **0.036** |

Векторный артефакт выигрывает 3 строки top-1 и 4 строки top-3, не теряя ни одной строки верной
тишины; см. `eval/experiments.md`, эксп. 17.

## Какой артефакт реально скорит

| Ситуация | Артефакт |
| --- | --- |
| `VECTOR_SEARCH=true`, эмбеддер `tfidf` | `reranker.tfidf.json` |
| `VECTOR_SEARCH=false` | `reranker.json` |
| `SEMANTIC_RERANKER=lexical` (при включённых векторах) | `reranker.json`, векторы остаются только кандидатами — откат без пересборки образа |
| эмбеддер без своего артефакта (например e5 в образе) | `reranker.json` + предупреждение в лог |

Фактический выбор виден в `GET /health/ready` → `vector_search.reranker` и `vector_search.scoring`.

## Снятый с эксплуатации артефакт

До 2026-10-23 в `reranker.json` лежал файл, который текущий тренер не воспроизводит: 29277 пар,
пороги `ok=0.840 / review=0.130`, ECE 0.102. На своей рабочей точке он давал 87.3 % top-1 при
74.1 % верной тишины — то есть был не лучше пересборки, а разговорчивее её: при одинаковой тишине
81.5 % оба дают 84.9 / 91.0 / 83.6 (см. `eval/experiments.md`, эксп. 18). Файл заменён
воспроизводимой пересборкой; старый доступен из истории:

```bash
git show 48c0fb1:ml/artifacts/reranker.json > /tmp/reranker-2026-09.json   # sha256 ce0fb5f2…
```

## Пересборка

```bash
python scripts/build_model.py --models-dir models                     # лексика + эмбеддер
python scripts/build_model.py --dense --gate --models-dir models      # + векторный, с допуском
cp models/reranker.json models/reranker.tfidf.json models/dense-gate.json ml/artifacts/
```

`--gate` переобучает лексическую базу тем же кодом (≈ 3 мин) и удаляет векторный артефакт, если
тот не прошёл допуск. После пересборки обновите sha256 в таблице выше — иначе упадёт
`api/tests/test_shipped_artifacts.py`.
