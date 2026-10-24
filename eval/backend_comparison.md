> **Исторический отчёт (2026-10-10).** Числа сняты на тогдашнем лексическом артефакте
> (`reranker.json`, sha `ce0fb5f2…`), который с тех пор снят с эксплуатации (эксп. 19);
> для `catalog` сейчас 84.3 / 89.8 / 85.2 (`eval/eval_report.md`). Сравнение режимов
> сохраняет силу как вывод о цене семантики, но не как текущие метрики.

# Сравнение бэкендов `catalog` и `qdrant` на размеченном наборе

**Строк:** 274 · **Разметка:** `eval/labeled_examples.csv` (измерительный набор, обучение не затрагивалось)

## Метрики

| Метрика | catalog | qdrant |
| --- | ---: | ---: |
| Top-1 (примеры с кодом) | 83.7% | 74.1% |
| Top-1 по всем примерам | 85.4% | 83.2% |
| Верный `not_found` | 88.0% | 97.2% |
| Латентность p50, мс | 38.5 | 170.6 |
| Латентность p95, мс | 63.4 | 195.8 |

## Дельты по проблемным срезам

| Срез | Всего | qdrant исправил | qdrant не изменил |
| --- | ---: | ---: | ---: |
| catalog промолчал, хотя код есть | 14 | 1 | 13 |
| catalog выдумал подсказку вне перечня | 13 | 10 | 3 |

| Вердикт | Строк |
| --- | ---: |
| both_miss | 29 |
| both_ok | 217 |
| fixed | 11 |
| regression | 17 |

## Строки, где изменился вердикт

| Запрос | Ожидалось | catalog | qdrant | Вердикт |
| --- | --- | --- | --- | --- |
| USB-флеш накопитель | (not_found) | 28.22.18.261 (review) | (not_found) (not_found) | fixed |
| Гриль угольный | (not_found) | 27.51.28 (review) | (not_found) (not_found) | fixed |
| Пульт дистанционного управления | (not_found) | 26.51.20.130 (review) | (not_found) (not_found) | fixed |
| Робот-пылесос | (not_found) | 28.99.39.200 (review) | (not_found) (not_found) | fixed |
| Свитер трикотажный мужской | 14.3 | (not_found) (not_found) | 14.3 (review) | fixed |
| Смарт-часы | (not_found) | 26.12.3 (review) | (not_found) (not_found) | fixed |
| Сушилка для обуви | (not_found) | 15.2 (review) | (not_found) (not_found) | fixed |
| Тележка гидравлическая | (not_found) | 28.22.14.159 (review) | (not_found) (not_found) | fixed |
| Фитнес-браслет | (not_found) | 32.30.14.120 (review) | (not_found) (not_found) | fixed |
| Шариковая ручка | (not_found) | 28.15.10 (review) | (not_found) (not_found) | fixed |
| Щётка для обуви | (not_found) | 15.2 (review) | (not_found) (not_found) | fixed |
| Блокнот на пружине | 17.23.13.191 | 17.23.13.191 (review) | (not_found) (not_found) | regression |
| Зарядная станция для электромобилей | 27.11.50.120 | 27.11.50.120 (review) | 27.90.11.900 (review) | regression |
| Каток дорожный самоходный | 28.92.24 | 28.92.24.120 (review) | (not_found) (not_found) | regression |
| Куртка зимняя мужская | 14.1 | 14.1 (review) | 32.99.11.140 (review) | regression |
| Медицинский морозильник | 28.25.13.110 | 28.25.13.110 (review) | 28.25.13.116 (review) | regression |
| Микрофон динамический | 26.40.41.000 | 26.40.41.000 (review) | (not_found) (not_found) | regression |
| Мобильный телефон Nokia кнопочный | 26.30.22 | 26.30.22 (review) | (not_found) (not_found) | regression |
| Носки хлопковые мужские | 14.1 | 14.1 (review) | 14.19.32.130 (review) | regression |
| Перчатки рабочие тканевые | 14.1 | 14.1 (review) | (not_found) (not_found) | regression |
| Петли дверные | 25.72.14.120 | 25.72.14.120 (review) | (not_found) (not_found) | regression |
| Платье женское летнее | 14.1 | 14.1 (review) | 32.99.11.140 (review) | regression |
| Покрышка грузовая | 22.11.1 | 22.11.1 (review) | (not_found) (not_found) | regression |
| Светодиодная лампа E27 12Вт | 27.40.1 | 27.40.1 (review) | (not_found) (not_found) | regression |
| Ситец набивной | 13.2 | 13.2 (review) | (not_found) (not_found) | regression |
| Смартфон Samsung Galaxy A54 | 26.30.22 | 26.30.22 (review) | (not_found) (not_found) | regression |
| Спирты этиловые | 20.14.2 | 20.14.2 (review) | (not_found) (not_found) | regression |
| Тетрадь школьная 12 листов | 17.23.13.194 | 17.23.13.194 (review) | (not_found) (not_found) | regression |

> Отчёт информационный: он не входит в CI-гейт и не меняет ни модель,
> ни пороги. Решение по режиму по умолчанию фиксируется в `eval/experiments.md`.
