"""
genai_title_with_keywords.py
============================

Скрипт для автоматического создания заголовка с обязательным включением
ключевых слов, извлечённых из заданного текста.

Алгоритм работы:
1. Читает исходный текст из файла.
2. Извлекает ключевые слова с помощью YAKE.
3. Формирует промпт для модели суммаризации mT5.
4. Генерирует краткий заголовок, который включает извлечённые ключевые слова.
5. Проверяет, все ли ключевые слова использованы, и при необходимости
   добавляет недостающие.

Запуск:
    python genai_title_with_keywords.py --infile input.txt --lang ru --num_keywords 5
"""

import argparse
import re
from pathlib import Path
import yake
from transformers import pipeline

# ---------------------------------------------------------------------
# Конфигурация по умолчанию для модели и генерации заголовка
# ---------------------------------------------------------------------
DEFAULT_SUMM_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
DEFAULT_MAX_LEN = 32
DEFAULT_MIN_LEN = 6
REPEAT_TRIES = 3


def read_text(path: Path) -> str:
    """
    Читает текст из файла и приводит его к «чистому» виду.
    """
    text = path.read_text(encoding="utf_8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keywords(text: str, lang: str = "ru", keywords_num: int = 6):
    """
    Извлекает ключевые слова из текста с помощью YAKE.
    Возвращает список уникальных токенов.
    """
    kw_extractor = yake.KeywordExtractor(lan=lang, n=1, dedupLim=0.9, top=keywords_num * 3)
    candidates = kw_extractor.extract_keywords(text)
    cleaned = []
    seen = set()
    for kw, _ in sorted(candidates, key=lambda x: x[1]):
        token = kw.strip().lower()
        token = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ\- ]+", "", token)
        token = re.sub(r"\s+", " ", token).strip(" -")
        if not token or token in seen or len(token) < 2:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= keywords_num:
            break
    return cleaned


def build_prompt(keywords, lang: str = "ru"):
    """
    Формирует инструкцию (промпт) для модели суммаризации.
    Усилено: запрет на скобки и «списки в конце».
    """
    kw_str = ", ".join(keywords)
    if lang.startswith("ru"):
        return (
            f"Создай заголовок, включив слова: {kw_str}. "
            f"Верни только один короткий заголовок (6–12 слов), без кавычек. "
            f"Не используй никакие скобки ((), [], {{}}) и не выводи перечисление слов в конце; "
            f"встрой слова органично в саму фразу."
        )
    else:
        return (
            f"Create a headline that includes the words: {kw_str}. "
            f"Return only one short headline (6–12 words), no quotes. "
            f"Do not use any brackets and do not append a list of words at the end; "
            f"integrate the words naturally into the sentence."
        )


def summarize_title(body_text: str,
                    prompt: str,
                    model_name: str = DEFAULT_SUMM_MODEL,
                    min_len: int = DEFAULT_MIN_LEN,
                    max_len: int = DEFAULT_MAX_LEN) -> str:
    """
    Генерирует заголовок на основе исходного текста и промпта.
    """
    summarizer = pipeline("summarization", model=model_name)
    inp = f"{prompt}\n\nтекст: {body_text}"
    out = summarizer(
        inp,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    title = out[0]["summary_text"]
    title = postprocess_title(title)
    return title


def postprocess_title(title: str) -> str:
    """
    Лёгкий санитайзер заголовка:
    - обрезает пробелы/кавычки,
    - схлопывает множественные пробелы,
    - удаляет финальные списки в скобках: (...), [...], {...}.
    """
    t = title.strip().strip("«»\"'“”")
    t = re.sub(r"\s+", " ", t)

    # удалить любые хвостовые скобочные вставки
    t = re.sub(r"\s*[\(\[\{][^)\]\}]{0,200}[\)\]\}]\s*$", "", t).strip()

    # убрать лишние: «— ,» и двойные знаки
    t = re.sub(r"\s+—\s*,", " —", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ", ", t)
    return t


def coverage_score(title: str, keywords: list[str]):
    """
    Оценивает покрытие ключевых слов (строгое вхождение по подстроке).
    """
    t = " " + title.lower() + " "
    missed = []
    hit = 0
    for kw in keywords:
        kw_l = kw.lower()
        if f" {kw_l} " in t or kw_l in t:
            hit += 1
        else:
            missed.append(kw)
    cov = 0.0 if not keywords else hit / len(keywords)
    return cov, missed


def compose_from_keywords_inline(keywords: list[str], lang: str = "ru") -> str:
    """
    Собирает простой читабельный заголовок из самих ключей,
    без скобок. Гарантирует вхождение всех ключевых слов.
    """
    if not keywords:
        return ""

    k = [x.strip() for x in keywords if x.strip()]
    first = k[0].capitalize()
    tail = k[1:]

    if lang.startswith("ru"):
        if len(tail) >= 3:
            # это не грамматически идеальный генератор, но гарантирует отсутствие скобок
            core = []
            # первые два как определение + существительное
            core.append(" ".join(tail[0:2]))
            # остальное склеим через запятую, иногда добавив предлог
            rest = []
            for i, tok in enumerate(tail[2:]):
                if i == 0 and not re.search(r"\bв\b|\bдля\b|\bна\b", tok):
                    rest.append(f"в {tok}")
                else:
                    rest.append(tok)
            phrase = ", ".join([c for c in [core[0]] + rest if c])
            return f"{first} — {phrase}".strip(" —,")
        else:
            return f"{first} — {', '.join(tail)}".strip(" —,")
    else:
        # generic EN-ish joiner
        return f"{first} — {', '.join(tail)}".strip(" —,")


def integrate_missing_inline(title: str, missed: list[str], all_keywords: list[str], lang: str = "ru") -> str:
    """
    Если модель что-то не включила, стратегия:
    1) Если заголовок пустой/слишком общий или содержит скобки — полностью
       пересобрать из ключей через compose_from_keywords_inline().
    2) Иначе — добавить через « — »/«: » компактной фразой без скобок.
    """
    t = title.strip()
    if not t or re.search(r"[\(\[\{][^)\]\}]{0,200}[\)\]\}]\s*$", t):
        return compose_from_keywords_inline(all_keywords, lang=lang)

    if not missed:
        # ничего не потеряно — вернём санитайзнутый вариант
        return postprocess_title(t)

    add = ", ".join(missed)
    # Если уже есть двоеточие/тире — добавим после запятой
    if " — " in t or ":" in t:
        return postprocess_title(f"{t}, {add}")
    # Иначе добавим через « — »
    return postprocess_title(f"{t} — {add}")


def main():
    """
    Точка входа CLI:
    * парсит аргументы командной строки,
    * извлекает ключевые слова,
    * генерирует заголовок с включением этих слов,
    * выводит метрики покрытия.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="Путь к исходному тексту")
    ap.add_argument("--lang", type=str, default="ru", help="Язык для извлечения ключевых слов (yake)")
    ap.add_argument("--num_keywords", type=int, default=6, help="Сколько ключевых слов извлекать")
    ap.add_argument("--model", type=str, default=DEFAULT_SUMM_MODEL, help="HF-модель для summarization")
    ap.add_argument("--min_coverage", type=float, default=0.8, help="Порог метрики покрытия ключевых слов")
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Макс длина заголовка (токены)")
    ap.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Мин длина заголовка (токены)")
    args = ap.parse_args()

    text = read_text(Path(args.infile))
    keywords_list = keywords(text, lang=args.lang, keywords_num=args.num_keywords)
    if not keywords_list:
        raise SystemExit("Не удалось извлечь ключевые слова — проверьте входной текст.")

    base_prompt = build_prompt(keywords_list, lang=args.lang)

    title = ""
    missed = keywords_list[:]
    cov = 0.0

    for attempt in range(1, REPEAT_TRIES + 1):
        # Первая/очередная генерация
        title = summarize_title(
            body_text=text,
            prompt=base_prompt if attempt == 1 else (
                build_prompt(keywords_list, lang=args.lang)
                + " Используй каждое слово дословно. Не используй скобки и не ставь все слова в конец."
            ),
            model_name=args.model,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        cov, missed = coverage_score(title, keywords_list)

        # если модель всё-таки вывела скобки — вычистим и проверим ещё раз
        cleaned = postprocess_title(title)
        if cleaned != title:
            title = cleaned
            cov, missed = coverage_score(title, keywords_list)

        if cov >= args.min_coverage and not re.search(r"[\(\[\{].*[\)\]\}]$", title):
            break

    # Если всё ещё ниже порога — встроим недостающие слова
    if cov < args.min_coverage or re.search(r"[\(\[\{].*[\)\]\}]$", title):
        title = integrate_missing_inline(title, missed, keywords_list, lang=args.lang)
        cov, missed = coverage_score(title, keywords_list)

    # ====== Вывод ======
    print("\n=== ключевые слова ===")
    print(", ".join(keywords_list))
    print("\n=== промпт ===")
    print(base_prompt)
    print("\n=== заголовок ===")
    print(title)
    print("\n=== метрика ===")
    used = len(keywords_list) - len(missed)
    total = len(keywords_list)
    print(f"покрытие ключевых слов: {used}/{total} = {used / total:.0%}")
    if missed:
        print("не использованы:", ", ".join(missed))
    else:
        print("все ключевыые слова учтены.")


if __name__ == "__main__":
    main()