"""
genai_batch_titles.py
=====================
Скрипт для пакетной обработки тем слайдов с итеративным сокращением текста до 30 слов.
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import yake
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Конфигурация
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
DEFAULT_MAX_LEN = 128
REPEAT_TRIES = 3
MAX_WORDS_LIMIT = 30 

def read_topics(path: Path) -> list[str]:
    """Читает файл построчно."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    lines = content.split('\n')
    topics = [line.strip() for line in lines if line.strip()]
    return topics

def extract_keywords_from_title(text: str, lang: str = "ru", keywords_num: int = 3) -> list[str]:
    """Извлекает ключевые слова из заголовка."""
    kw_extractor = yake.KeywordExtractor(lan=lang, n=1, dedupLim=0.9, top=10, windowsSize=1)
    candidates = kw_extractor.extract_keywords(text)
    
    cleaned = []
    seen = set()
    sorted_candidates = sorted(candidates, key=lambda x: x[1])
    
    for kw, _ in sorted_candidates:
        token = kw.strip().lower()
        token = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ\- ]+", "", token)
        if not token or token in seen or len(token) < 3:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= keywords_num:
            break
            
    if not cleaned:
        words = re.findall(r"\w+", text)
        cleaned = [w.lower() for w in words if len(w) > 4][:keywords_num]
        
    return cleaned

def build_prompt(topic: str, keywords: list[str], lang: str = "ru") -> list[dict]:
    """
    Формирует промпт с жестким требованием к языку и длине.
    """
    kw_str = ", ".join(keywords)
    
    if lang.startswith("ru"):
        system_prompt = "Ты — эксперт по созданию презентаций и копирайтингу. Ты пишешь ТОЛЬКО на русском языке."
        user_prompt = (
            f"ЗАДАЧА:\n"
            f"Напиши информативное описание для этого слайда. "
            f"ВНИМАНИЕ: Используй ТОЛЬКО чистый русский язык. Исключи любые английские или другие иностранные слова, всегда используй их русские эквиваленты."
            f"Объем: ЧУТЬ МЕНЬШЕ {MAX_WORDS_LIMIT} слов.\n"
            f"ОБЯЗАТЕЛЬНО включи в текст эти ключевые слова: {kw_str}.\n"
            f"Не повторяй заголовок слово в слово. Раскрой суть темы.\n"
            f"Тема слайда: \"{topic}\"\n\n"
        )
    else:
        system_prompt = "You are an expert in presentation design and copywriting."
        user_prompt = (
            f"TASK:\n"
            f"Write an informative description for this slide. "
            f"Length: strictly no more than {MAX_WORDS_LIMIT} words.\n"
            f"You MUST include these keywords: {kw_str}.\n"
            f"Do not just repeat the title. Expand on the core idea."
            f"Slide Topic: \"{topic}\"\n\n"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def generate_description(model, tokenizer, messages: list[dict], max_len: int = DEFAULT_MAX_LEN) -> str:
    """Генерирует текст."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_len,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return postprocess_text(response)

def postprocess_text(text: str) -> str:
    """Чистка текста от лишних символов и фраз модели."""
    t = text.strip().strip("«»\"'“”")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(Конечно|Вот описание|Описание слайда|Суть слайда)[:\.]?\s*", "", t, flags=re.IGNORECASE)
    return t

def coverage_score(text: str, keywords: list[str]):
    """Проверяет наличие ключевых слов."""
    t = " " + text.lower() + " "
    missed = []
    hit = 0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in t: 
            hit += 1
        else:
            missed.append(kw)
    cov = 0.0 if not keywords else hit / len(keywords)
    return cov, missed

def enforce_length_limit(model, tokenizer, original_text: str, limit: int, attempt_limit: int = 3) -> str:
    """
    Принудительно сокращает текст, используя модель (итеративное переписывание).
    """
    
    length_messages = [
        {"role": "system", "content": "Ты — профессиональный редактор. Твоя единственная задача — сократить текст до требуемой длины, сохраняя его смысл и тон."},
        {"role": "user", "content": f"Сократи следующий текст СТРОГО до {limit} слов. Текст: \"{original_text}\""}
    ]
    
    current_desc = original_text
    
    for attempt in range(1, attempt_limit + 1):
        word_count = len(current_desc.split())
        
        if word_count <= limit:
            return current_desc 
        
        if attempt > 1:
            print(f"    -> Попытка {attempt}: Повторное сокращение (Текущая длина: {word_count} слов)...")
            length_messages.append({"role": "assistant", "content": current_desc})
            length_messages.append({"role": "user", "content": f"Ты не справился с задачей. Повтори сокращение СТРОЖЕ. Текст должен быть НЕ БОЛЕЕ {limit} слов. Сохрани оригинальный смысл."})

        current_desc = generate_description(model, tokenizer, length_messages)

        word_count = len(current_desc.split())
        
        if word_count <= limit:
            return current_desc 

    print(f"    -> Превышено число попыток сокращения. Финальная длина: {word_count} слов.")
    return current_desc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True, help="Путь к файлу со списком тем")
    ap.add_argument("--outfile", type=str, default="slides_output.csv", help="Файл результата")
    ap.add_argument("--lang", type=str, default="ru", help="Язык")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Модель HF")
    ap.add_argument("--show_table", action="store_true", help="Вывести результаты в терминал в формате 'Тема ----- Результат'")
    args = ap.parse_args()

    print(f"Загрузка модели {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype="auto", 
            device_map="auto"
        )
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    infile = Path(args.infile)
    topics = read_topics(infile)
    if not topics:
        print(f"Файл {infile} пуст или не найден.")
        return
        
    print(f"Найдено тем для обработки: {len(topics)}")
    
    results = []

    for idx, topic in enumerate(topics, 1):
        print(f"Processing {idx}/{len(topics)}...") 
        
        kws = extract_keywords_from_title(topic, lang=args.lang, keywords_num=3)
        
        messages = build_prompt(topic, kws, lang=args.lang)
        best_desc = ""
        best_cov = -1.0
        
        for attempt in range(1, REPEAT_TRIES + 1):
            description = generate_description(model, tokenizer, messages)
            cov, missed = coverage_score(description, kws)
            
            if cov > best_cov:
                best_cov = cov
                best_desc = description

            if cov >= 0.99:
                break
            else:
                missing_str = ", ".join(missed)
                messages.append({"role": "assistant", "content": description})
                messages.append({"role": "user", "content": f"Перепиши описание, чтобы обязательно добавить слова: {missing_str}. Сохрани длину НЕ БОЛЕЕ {MAX_WORDS_LIMIT} слов."})

        final_desc = best_desc

        cov, missed = coverage_score(final_desc, kws)
        if len(missed) > 0:
            final_desc += f" (Ключевые слова: {', '.join(missed)})"
        
        print(f"  -> Исходная длина: {len(final_desc.split())} слов.")
        print(f"  -> Ключевые слова: {kws}.")
        final_desc = enforce_length_limit(model, tokenizer, final_desc, MAX_WORDS_LIMIT, REPEAT_TRIES)
                
        word_count = len(final_desc.split())
        is_length_ok = word_count <= MAX_WORDS_LIMIT
        
        cov, missed = coverage_score(final_desc, kws)
        kw_ok = len(missed) == 0

        metric_status = ""
        metric_status += f"длина ≤ {MAX_WORDS_LIMIT} слов ({'OK' if is_length_ok else f'НЕ OK, {word_count} слов'})"
        metric_status += ", ключевые слова "
        metric_status += "есть" if kw_ok else f"отсутствуют ({len(missed)} шт.)"
        
        
        results.append({
            "slide_title": topic,
            "content": final_desc,
            "metric": metric_status
        })
        print(f"  -> {final_desc}")
        print(f"  -> Метрика: {metric_status}")

    df = pd.DataFrame(results)
    outfile = Path(args.outfile)
    df.to_csv(outfile, index=False, encoding="utf-8")
    print(f"\nДанные сохранены в файл: {outfile}")

    if args.show_table:
        print("\n" + "="*80)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*80)
        for item in results:
            print(f"{item['slide_title']}\n----- {item['content']}")
            print(f"Метрика: {item['metric']}")
            print("-" * 50)
        print("="*80 + "\n")

if __name__ == "__main__":
    main()