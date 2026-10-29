from __future__ import annotations
"""
Справка для вкладки «Справка» веб-интерфейса.
Подсказки к отдельным параметрам — в params_catalog.py (всплывают при наведении).
"""

import html
from textwrap import dedent


class HelpContent:
    @staticmethod
    def _md_to_html(md: str) -> str:
        """ Очень простой конвертер под наш текст: заголовки, списки, абзацы. """
        out: list[str] = []
        in_list = False
        for ln in md.strip().splitlines():
            is_item = ln.startswith("- ")
            if in_list and not is_item:
                out.append("</ul>")
                in_list = False
            if ln.startswith("# "):
                out.append(f"<h2>{html.escape(ln[2:])}</h2>")
            elif ln.startswith("## "):
                out.append(f"<h3>{html.escape(ln[3:])}</h3>")
            elif is_item:
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{html.escape(ln[2:])}</li>")
            elif ln.strip():
                out.append(f"<p>{html.escape(ln)}</p>")
        if in_list:
            out.append("</ul>")
        return "\n".join(out)

    @staticmethod
    def get_sections() -> dict[str, str]:
        md_intro = dedent("""
        # Коротко: как запустить обучение
        - 1. «Модель и источник»: стандартная модель (линейка → задача → размер) или свой файл .pt.
        - 2. Датасет: data.yaml (детекция, сегментация, pose, OBB) или папка (классификация).
        - 3. «Параметры обучения»: выберите профиль и нажмите «Подставить» — или задайте значения вручную.
        - 4. «Проверить на датасете» — быстрый прогон без обучения: видно, что датасет и модель читаются.
        - 5. «Старт» — обучение идёт в отдельном процессе; ход — на вкладке «Прогресс».
        Наведите курсор на любое поле — появится подсказка, что делает параметр.

        ## Где результаты
        - <папка результатов>/<задача>/<ран>/weights/best.pt — лучшие веса, last.pt — последние.
        - results.csv и графики Ultralytics, train_metrics.jsonl (график прогресса), train.log (вывод процесса).
        """)

        md_finetune = dedent("""
        # Своя модель: дообучение и продолжение
        - Дообучение — новое обучение, начиная с весов вашего файла (например best.pt прошлого рана). Новая папка рана, текущие параметры формы. Подходит, когда добавили данные или классы.
        - Продолжить прерванное — выберите weights/last.pt прерванного рана: обучение продолжится с той же эпохи, с теми же параметрами, в той же папке.
        - Задача (классификация, детекция и т.д.) определяется из самого файла — кнопка «Проверить файл».
        - При дообучении на маленьком датасете помогает «Заморозить слоёв» = 10 и меньшая скорость обучения.
        """)

        md_tasks = dedent("""
        # Задачи и метрики
        - Классификация — класс всего изображения. Метрика: top1 (доля верных ответов). imgsz 224–384.
        - Детекция — рамки объектов. Метрика: mAP50-95 (основная), mAP50.
        - Сегментация — маски объектов. Метрика: mAP50-95 по маскам; памяти нужно больше, чем детекции.
        - Pose — ключевые точки (суставы, углы и т.п.). Метрика: mAP50-95 по точкам. В data.yaml нужны kpt_shape и flip_idx.
        - OBB — повёрнутые рамки (объекты под углом, вид сверху). Метрика: mAP50-95.
        """)

        md_tips = dedent("""
        # Частые ситуации
        - Нехватка видеопамяти (CUDA out of memory) — уменьшить batch, затем imgsz; профиль low_vram.
        - Метрика не растёт — проверить разметку и пути в data.yaml; ослабить аугментации; больше эпох.
        - Train-потери падают, а метрика val ухудшается — переобучение: больше данных, аугментации, dropout (классификация), меньше эпох.
        - «Стоп» завершает обучение после текущей эпохи с сохранением весов; «Прервать» — немедленно (last.pt — последней сохранённой эпохи).
        """)

        return {"intro": md_intro, "finetune": md_finetune, "tasks": md_tasks, "tips": md_tips}

    @staticmethod
    def get_html_sections() -> dict[str, str]:
        """ Разделы справки в HTML. """
        return {k: HelpContent._md_to_html(v) for k, v in HelpContent.get_sections().items()}

    @staticmethod
    def get_markdown() -> str:
        return "\n\n".join(HelpContent.get_sections().values())
