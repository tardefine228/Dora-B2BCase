import re
import pymorphy3


class SmartProductAnalyzer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

        self.categories = {
            "ноутбук": ["ноутбук", "лэптоп", "ультрабук", "нетбук"],
            "сервер": ["сервер", "blade-система"],
            "планшет": ["планшет", "pad"],
            "монитор": ["монитор", "дисплей", "экран"],
            "принтер": ["принтер", "мфу", "сканер"]
        }

        self.brands_to_remove = [
            "lenovo", "hp", "dell", "apple", "asus", "acer", "samsung",
            "xiaomi", "huawei", "sony", "lg", "canon", "epson"
        ]

        self.spec_markers = ["дюйм", "гб", "мгц", "ггц", "ядер", "тб", "ssd", "hdd"]

    def analyze(self, text: str) -> dict:
        if not text:
            return {"type": "unknown", "specs": "", "cleaned": ""}

        text_lower = text.lower()

        # 1. Определяем тип предмета
        detected_type = "товар общего назначения"
        for item_type, keywords in self.categories.items():
            if any(kw in text_lower for kw in keywords):
                detected_type = item_type
                break

        # Ищем паттерны: 15.6 дюймов, 8 ГБ, i5-12400 (как модель процессора)
        specs = []
        spec_pattern = r'(\d+\.?\d*)\s*(дюйм|гб|мгц|ггц|тб|мм|кг)'
        found_specs = re.findall(spec_pattern, text_lower)
        for val, unit in found_specs:
            specs.append(f"{val} {unit}")


        # 3. Формируем "чистый" запрос для векторного поиска
        cleaned_parts = []
        for word in text_lower.split():
            # Убираем артикулы (длинные наборы букв и цифр)
            if re.match(r'^[a-z0-9]{5,}$', word): continue
            # Убираем бренды устройств
            if word in self.brands_to_remove: continue

            p = self.morph.parse(word)
            cleaned_parts.append(p[0].normal_form)

        cleaned_text = " ".join(set(cleaned_parts))

        return {
            "type": detected_type,
            "specs": ", ".join(specs) if specs else "стандартные характеристики",
            "cleaned": cleaned_text
        }


if __name__ == "__main__":
    analyzer = SmartProductAnalyzer()

    test_cases = [
        """Планшет Samsung Galaxy Tab A8 10.5" WiFi 32GB""",
        "Жесткий диск SSD 2.5 SATA III 480GB Kingston",
        "Устройство ввода-вывода для формирования твердой копии документа на бумаге"
    ]

    for case in test_cases:
        result = analyzer.analyze(case)
        print(f"Исходный: {case}")
        print(f"Тип: {result['type']}")
        print(f"Характеристики: {result['specs']}")
        print(f"Для поиска: {result['cleaned']}\n")