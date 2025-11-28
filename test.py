"""
genai_title_with_keywords.py
============================

Скрипт для генерации описания для слайда презентации с обязательным
включением ключевых слов.

Алгоритм работы:
1. Получает тему слайда из файла.
2. Извлекает ключевые слова с помощью YAKE.
3. Формирует промпт для модели mT5.
4. Генерирует краткое описание (около 30 слов), включая ключевые слова.
5. Проверяет длину и наличие ключевых слов.
6. Сохраняет результат в CSV-файл.
7. Выводит результат в виде таблицы.

Запуск:
    python genai_title_with_keywords.py --infile input.txt --outfile result.csv
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import yake
from transformers import pipeline

# ---------------------------------------------------------------------
# Конфигурация по умолчанию для модели и генерации описания
# ---------------------------------------------------------------------
DEFAULT_SUMM_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
DEFAULT_MAX_LEN = 60
DEFAULT_MIN_LEN = 20
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

def build_prompt(topic: str, keywords: list[str], lang: str = "ru"):
    """
    Формирует инструкцию (промпт) для модели.
    """
    kw_str = ", ".join(keywords)
    if lang.startswith("ru"):
        return (
            f"Сгенерируй краткое описание для слайда на тему '{topic}', "
            f"включив в него ключевые слова: {kw_str}. "
            f"Описание должно быть длиной около 30 слов. "
            f"Не используй скобки и кавычки."
        )
    else:
        return (
            f"Generate a short description for a slide on the topic '{topic}', "
            f"including the keywords: {kw_str}. "
            f"The description should be about 30 words long. "
            f"Do not use brackets or quotes."
        )

def generate_description(prompt: str,
                         model_name: str = DEFAULT_SUMM_MODEL,
                         min_len: int = DEFAULT_MIN_LEN,
                         max_len: int = DEFAULT_MAX_LEN) -> str:
    """
    Генерирует описание на основе промпта.
    """
    summarizer = pipeline("summarization", model=model_name)
    out = summarizer(
        prompt,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    description = out[0]["summary_text"]
    description = postprocess_text(description)
    return description

def postprocess_text(text: str) -> str:
    """
    Лёгкий санитайзер текста:
    - обрезает пробелы/кавычки,
    - схлопывает множественные пробелы,
    - удаляет финальные списки в скобках: (...), [...], {...}.
    """
    t = text.strip().strip("«»\"'“”")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*[\(\[\{][^)\]\}]{0,200}[\)\]\}]\s*$", "", t).strip()
    t = re.sub(r"\s+—\s*,", " —", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ", ", t)
    return t


def coverage_score(text: str, keywords: list[str]):
    """
    Оценивает покрытие ключевых слов.
    """
    t = " " + text.lower() + " "
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


def compose_from_keywords(keywords: list[str], topic: str = "", lang: str = "ru") -> str:
    """
    Собирает описание из ключевых слов, если модель не справилась.
    """
    if not keywords:
        return ""

    kw_str = ", ".join(keywords)
    if lang.startswith("ru"):
        return f"Краткое описание для слайда на тему '{topic}': {kw_str}."
    else:
        return f"A short description for the slide on '{topic}': {kw_str}."


def integrate_missing(description: str, missed: list[str], all_keywords: list[str], topic: str, lang: str = "ru") -> str:
    """
    Если модель что-то не включила, добавляем недостающие слова.
    """
    d = description.strip()
    if not d or len(d.split()) < 10:
        return compose_from_keywords(all_keywords, topic=topic, lang=lang)

    if not missed:
        return postprocess_text(d)

    add = ", ".join(missed)
    return postprocess_text(f"{d}. Дополнительные ключевые слова: {add}")


def save_to_csv(slide_title: str, content: str, outfile: Path):
    """
    Сохраняет данные в CSV-файл.
    """
    df = pd.DataFrame([[slide_title, content]], columns=["slide_title", "content"])
    df.to_csv(outfile, index=False, encoding="utf-8")


def display_table(outfile: Path):
    """
    Выводит содержимое CSV-файла в виде таблицы.
    """
    try:
        df = pd.read_csv(outfile)
        print("\n" + df.to_string(index=False))
    except FileNotFoundError:
        print(f"Ошибка: файл {outfile} не найден.")


def main():
    """
    Точка входа CLI.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="Путь к файлу с темой слайда")
    ap.add_argument("--outfile", type=str, default="slide_description.csv", help="Путь к CSV файлу для сохранения результата")
    ap.add_argument("--lang", type=str, default="ru", help="Язык для извлечения ключевых слов (yake)")
    ap.add_argument("--num_keywords", type=int, default=5, help="Сколько ключевых слов извлекать")
    ap.add_argument("--model", type=str, default=DEFAULT_SUMM_MODEL, help="HF-модель для summarization")
    ap.add_argument("--min_coverage", type=float, default=0.8, help="Порог метрики покрытия ключевых слов")
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN, help="Макс длина описания (токены)")
    ap.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN, help="Мин длина описания (токены)")
    args = ap.parse_args()

    topic = read_text(Path(args.infile))
    keywords_list = keywords(topic, lang=args.lang, keywords_num=args.num_keywords)
    if not keywords_list:
        raise SystemExit("Не удалось извлечь ключевые слова — проверьте входной текст.")

    base_prompt = build_prompt(topic, keywords_list, lang=args.lang)

    description = ""
    missed = keywords_list[:]
    cov = 0.0

    for attempt in range(1, REPEAT_TRIES + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += " Используй все ключевые слова."

        description = generate_description(
            prompt=prompt,
            model_name=args.model,
            min_len=args.min_len,
            max_len=args.max_len,
        )
        cov, missed = coverage_score(description, keywords_list)
        
        cleaned = postprocess_text(description)
        if cleaned != description:
            description = cleaned
            cov, missed = coverage_score(description, keywords_list)

        if cov >= args.min_coverage:
            break

    if cov < args.min_coverage:
        description = integrate_missing(description, missed, keywords_list, topic, lang=args.lang)
        cov, missed = coverage_score(description, keywords_list)

    # ====== Сохранение и вывод ======
    outfile = Path(args.outfile)
    save_to_csv(topic, description, outfile)

    print(f"\nРезультат сохранён в: {outfile}")
    display_table(outfile)


if __name__ == "__main__":
    main()