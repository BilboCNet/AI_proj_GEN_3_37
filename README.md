
#  About

Код решения задачи GEN_3_37 выполняет следующее:
 
* Читает темы из файла input.txt
* Генерирует краткое описание (<=30 слов), включив ключевые слова.
* Проверяет длину.
* Сохраняет в CSV: slide_title, content, metric.
* Выводит таблицу(опционально).

# Installation 
```
git clone https://github.com/BilboCNet/AI_proj_GEN_3_37.git
cd AI_proj_GEN_3_37

pip install -r requirements.txt
```
# Quickstart

Для тестирования на синтетических темах и вывода таблицы в терминал(--show_table).
```
python genai_title_with_keywords.py --infile input.txt --lang ru --show_table
```
