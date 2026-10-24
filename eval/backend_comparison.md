# Сравнение бэкендов `catalog` и `qdrant` на размеченном наборе

**Строк:** 233 · **Разметка:** `eval/labeled_examples.csv` (измерительный набор, обучение не затрагивалось)

## Метрики

| Метрика | catalog | qdrant |
| --- | ---: | ---: |
| Top-1 (примеры с кодом) | 95.0% | 68.7% |
| Top-1 по всем примерам | 95.3% | 71.7% |
| Верный `not_found` | 96.9% | 90.6% |
| Латентность p50, мс | 33.2 | 4.8 |
| Латентность p95, мс | 46.0 | 6.5 |

## Дельты по проблемным срезам

| Срез | Всего | qdrant исправил | qdrant не изменил |
| --- | ---: | ---: | ---: |
| catalog промолчал, хотя код есть | 3 | 0 | 0 |
| catalog выдумал подсказку вне перечня | 1 | 0 | 1 |

| Вердикт | Строк |
| --- | ---: |
| both_miss | 9 |
| both_ok | 165 |
| fixed | 2 |
| regression | 57 |

## Строки, где изменился вердикт

| Запрос | Ожидалось | catalog | qdrant | Вердикт |
| --- | --- | --- | --- | --- |
| Свитер трикотажный мужской | 14.3 | 14.1 (review) | 14.3 (ok) | fixed |
| Труба алюминиевая | 24.42.26 | 32.20.13.161 (review) | 24.42.26 (ok) | fixed |
| USB-флеш накопитель | 26.20.2 | 26.20.2 (ok) | (not_found) (not_found) | regression |
| Аппарат ультразвуковой хирургический | 26.60.12.132 | 26.60.12.132 (ok) | 26.60.13.130 (ok) | regression |
| Беспроводная колонка | 26.40.42.110 | 26.40.42.110 (ok) | 26.30.22 (review) | regression |
| Блокнот на пружине | 17.23.13.191 | 17.23.13.191 (ok) | (not_found) (not_found) | regression |
| Болгарка угловая | 28.24.1 | 28.24.1 (ok) | (not_found) (not_found) | regression |
| Ботинки кожаные мужские | 15.2 | 15.2 (ok) | 32.30.12.110 (review) | regression |
| Веб-камера | 26.20.16 | 26.20.16 (ok) | (not_found) (not_found) | regression |
| Гантели разборные | 32.30.14.110 | 32.30.14.110 (ok) | (not_found) (not_found) | regression |
| Гарнитура для компьютера | 26.20.16 | 26.20.16 (ok) | (not_found) (not_found) | regression |
| Грабли садовые | 25.73.30 | 25.73.30 (ok) | (not_found) (not_found) | regression |
| Гриль угольный | 27.51.28 | 27.51.28 (review) | (not_found) (not_found) | regression |
| Двери металлические входные | 25.12.10.000 | 25.12.10.000 (ok) | (not_found) (not_found) | regression |
| Дрон квадрокоптер | 30.30.32.150 | 30.30.32.150 (ok) | (not_found) (not_found) | regression |
| Зарядная станция для электромобилей | 27.11.50.120 | 27.11.50.120 (review) | 27.90.11.900 (review) | regression |
| Камера велосипедная | 22.11.1 | 22.11.1 (ok) | (not_found) (not_found) | regression |
| Камера видеонаблюдения | 26.30.13 | 26.30.13 (ok) | 26.40.33.110 (review) | regression |
| Квадрокоптер | 30.30.32.150 | 30.30.32.150 (ok) | (not_found) (not_found) | regression |
| Коврик для йоги | 32.30.14.110 | 32.30.14.110 (review) | (not_found) (not_found) | regression |
| Компьютер настольный | 26.20.14 | 26.20.14 (ok) | (not_found) (not_found) | regression |
| Конструктор детский | 32.4 | 32.4 (ok) | (not_found) (not_found) | regression |
| Костюм мужской | 14.1 | 14.1 (ok) | (not_found) (not_found) | regression |
| Кресла-стулья с санитарным оснащением | 32.50.22.129 | 32.50.22.129 (review) | 32.50.50 (review) | regression |
| Кроссовки мужские | 15.2 | 15.2 (ok) | (not_found) (not_found) | regression |
| Куртка зимняя мужская | 14.1 | 14.1 (ok) | (not_found) (not_found) | regression |
| Лак паркетный | 20.30 | 20.30 (ok) | (not_found) (not_found) | regression |
| Лобзик электрический | 28.24.1 | 28.24.1 (ok) | (not_found) (not_found) | regression |
| МФУ лазерное | 26.20.18 | 26.20.18 (ok) | (not_found) (not_found) | regression |
| Мангал | 27.51.28 | 27.51.28 (ok) | (not_found) (not_found) | regression |
| Масло сливочное 72,5% | 10.51.30.100 | 10.51.30.100 (ok) | 10.51.30.110 (ok) | regression |
| Медицинский морозильник | 28.25.13.110 | 28.25.13.110 (review) | 32.50.50 (review) | regression |
| Миксер ручной | 27.51.21.120 | 27.51.21.120 (ok) | (not_found) (not_found) | regression |
| Мышь компьютерная | 26.20.16 | 26.20.16 (ok) | (not_found) (not_found) | regression |
| Настольный футбол | 32.4 | 32.4 (ok) | (not_found) (not_found) | regression |
| Носки хлопковые мужские | 14.1 | 14.1 (ok) | (not_found) (not_found) | regression |
| Ноутбук Lenovo IdeaPad | 26.20.11 | 26.20.11 (ok) | (not_found) (not_found) | regression |
| Окна пластиковые | 25.12.10.000 | 25.12.10.000 (ok) | (not_found) (not_found) | regression |
| Пазл | 32.4 | 32.4 (ok) | (not_found) (not_found) | regression |
| Перфоратор Bosch | 28.24.1 | 28.24.1 (ok) | (not_found) (not_found) | regression |
| Перфоратор Makita | 28.24.1 | 28.24.1 (ok) | (not_found) (not_found) | regression |
| Перчатки рабочие тканевые | 14.1 | 14.1 (ok) | 22.19.60.119 (review) | regression |
| Планшет Samsung Galaxy Tab | 26.20.11 | 26.20.11 (ok) | (not_found) (not_found) | regression |
| Платье женское летнее | 14.1 | 14.1 (ok) | (not_found) (not_found) | regression |
| Принтер лазерный | 26.20.16 | 26.20.16 (ok) | (not_found) (not_found) | regression |
| Сапоги детские | 15.2 | 15.2 (ok) | 32.50.22.150 (ok) | regression |
| Сварочный аппарат инверторный | 27.90.31.110 | 27.90.31.110 (review) | (not_found) (not_found) | regression |
| Сейф офисный | (not_found) | (not_found) (not_found) | 28.23.23 (review) | regression |
| Ситец набивной | 13.2 | 13.2 (ok) | 13.20.46 (review) | regression |
| Скейтборд | 32.30.14.110 | 32.30.14.110 (ok) | (not_found) (not_found) | regression |
| Труба пластиковая канализационная | 22.21.2 | 22.21.2 (ok) | (not_found) (not_found) | regression |
| Туфли женские кожаные | 15.2 | 15.2 (ok) | (not_found) (not_found) | regression |
| Утюг с парогенератором | 27.51.23.130 | 27.51.23.130 (ok) | (not_found) (not_found) | regression |
| Шпатель малярный | 25.73.30 | 25.73.30 (ok) | (not_found) (not_found) | regression |
| Шредер для бумаги | 28.23.23 | 28.23.23 (ok) | (not_found) (not_found) | regression |
| Шуба норковая | 14.20 | 14.20 (ok) | (not_found) (not_found) | regression |
| Шуруповерт аккумуляторный | 28.24.1 | 28.24.1 (ok) | (not_found) (not_found) | regression |
| Щётка для обуви | (not_found) | (not_found) (not_found) | 15.2 (review) | regression |
| ноутбук для работы | 26.20.11 | 26.20.11 (ok) | (not_found) (not_found) | regression |

> Отчёт информационный: он не входит в CI-гейт и не меняет ни модель,
> ни пороги. Решение по режиму по умолчанию фиксируется в `eval/experiments.md`.
