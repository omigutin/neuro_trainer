from __future__ import annotations

from textwrap import dedent
import html

class HelpContent:
    @staticmethod
    def _md_to_html(md: str) -> str:
        """Очень простой конвертер под наш текст (заголовки/списки/код/жирный)."""
        lines = md.strip().splitlines()
        out = []
        for ln in lines:
            if ln.startswith("### "):
                out.append(f"<h3>{html.escape(ln[4:])}</h3>")
            elif ln.startswith("## "):
                out.append(f"<h3>{html.escape(ln[3:])}</h3>")
            elif ln.startswith("# "):
                out.append(f"<h2>{html.escape(ln[2:])}</h2>")
            elif ln.startswith("- "):
                out.append(f"<li>{html.escape(ln[2:])}</li>")
            elif ln.strip() == "":
                out.append("<br/>")
            else:
                out.append(f"<p>{html.escape(ln)}</p>")
        # склеим соседние <li> в <ul>
        html_joined = "\n".join(out)
        html_joined = html_joined.replace("</li>\n<li>", "</li><li>")
        html_joined = html_joined.replace("<li>", "<ul><li>", 1).replace("</li>", "</li></ul>", 1) if "<li>" in html_joined else html_joined
        return html_joined

    @staticmethod
    def get_sections() -> dict[str, str]:
        md_intro = dedent("""
        # Коротко: как выбирать настройки
        1) Выбери модель и тип (auto определится сам).
        2) Укажи датасет (папка или data.yaml / Roboflow URL).
        3) Профиль: fast / quality / low_vram.
        4) Гиперпараметры: epochs/patience/batch/imgsz. Для CLS — одно число (квадрат). Для DET/SEG — число или HxW.
        5) Валидация — быстрый чек перед стартом (кнопка «Валидация»).
        6) Старт — обучение; смотри вкладку Progress (лог, графики).

        Где искать результаты:
        - внутри project_dir/task/name: results.csv, train_metrics.jsonl, final_metrics.json, final_metrics_extra.csv.
        """)

        md_cls = dedent("""
        # Классификация (Classifier)
        Когда картинка относится к одному классу целиком.
        Ключевые поля:
        - imgsz (одно число): итоговый квадрат.
          224 — очень быстро; 320 — баланс; 384 — качество (нужна VRAM).
        - batch: сколько картинок за раз.
        - epochs: сколько «кругов» обучения.
        - patience: ранний стоп, если качество не растёт.

        Полезные приёмы:
        - auto_augment="randaugment" — «умная мешалка»; помогает не переобучаться.
        - erasing (0..1) — случайно «затираем» кусок; модель учится быть устойчивее.
        - label_smoothing (0..1) — слегка «размываем» метку → меньше переобучение.
        - dropout (0..1) — отключаем часть нейронов на тренировке.

        Метрика:
        - top1 — доля правильных ответов; это основной ориентир.
        """)

        md_det = dedent("""
        # Детекция (Detector)
        Когда нужно найти объекты и их коробки на изображении.

        Важное:
        - imgsz: 640 — «классика», 800 — детальнее (медленнее), 512 — экономия VRAM.
        - rect=True — собираем батчи со схожими пропорциями → меньше паддинга.
        - multi_scale=True — меняем масштаб на лету → устойчивость к расстоянию.
        - batch/epochs/patience — как в CLS, но учится дольше.

        Метрики:
        - mAP50-95 (основная), mAP50 и mAP75 — для ранних оценок.
        """)

        md_seg = dedent("""
        # Сегментация (Segmentator)
        Пиксельные маски объектов (точные контуры).

        Важное:
        - imgsz: требования к памяти выше, чем у детекции.
        - rect/multi_scale — по смыслу как в детекции.
        - batch — часто приходится снижать.

        Метрики:
        - mAP по маскам (рядом с box в отчёте Ultralytics).
        """)

        md_train = dedent("""
        # Параметры train() — простым языком
        Основное: data, epochs, batch, imgsz, patience, device, workers, seed/deterministic.

        • data — путь к датасету/`data.yaml` или Roboflow URL. Ошибки путей — 80% проблем.
        • epochs — сколько «кругов» обучения. Больше — дольше и обычно лучше (при хорошем датасете).
        • batch — размер пачки. Упирается в VRAM. При OOM — сначала уменьшаем batch.
        • imgsz — размер входа. CLS — одно число (делаем квадрат). DET/SEG — число или HxW. Мы округлим до кратности 32.
        • patience — «терпение» без улучшений; помогает не тратить время зря.
        • device — auto/cuda/mps/cpu. `auto` сам выберет.
        • workers — потоки загрузки данных. Обычно 4–8.
        • seed/deterministic — для воспроизводимости.

        Детектор/Сегментатор:
        • rect — батчи с похожими пропорциями; меньше паддинга.
        • multi_scale — меняем масштаб на лету; полезно для робастности.

        Аугментации (частично игнорируются в CLS):
        • hsv_h/s/v — цветовые сдвиги; помогают при «трудных» освещениях.
        • degrees/translate/scale/shear/perspective — геометрические сдвиги.
        • flipud/fliplr — перевороты; аккуратно с «есть право-лево».
        • mosaic/mixup/copy_paste — синтетические миксы; часто улучшают детектор/сегментатор.
        • auto_augment/erasing/label_smoothing/dropout — регуляризация для CLS.
        """)

        md_val = dedent("""
        # Параметры val() — простым языком
        Валидация запускается автоматически внутри train() на каждом цикле.
        Кнопка «Валидация» в GUI делает отдельный быстрый прогон.

        Важно:
        • data — путь к датасету/`data.yaml` (или Roboflow URL).
        • imgsz — должен соответствовать тому, как тренируем (мы приведём к кратности 32).
        • device, workers — как в train().
        • В отчётах для CLS — top1/top5; для DET/SEG — mAP50-95, mAP50, mAP75 и др.

        Типичные проверки:
        • Структура датасета корректна? (папки/файлы, классы, пути в `data.yaml`)
        • Данных хватает? Если нет — риск переобучения (смотри на разрыв train/val).
        • Метрики «не двигаются»: возможно, неверная разметка или слишком агрессивные аугментации.
        """)

        return {
            "intro": md_intro,
            "classifier": md_cls,
            "detector": md_det,
            "segmentator": md_seg,
            "train_params": md_train,
            "val_params": md_val,
        }

    @staticmethod
    def get_markdown() -> str:
        s = HelpContent.get_sections()
        return "\n\n".join([s["intro"], s["classifier"], s["detector"], s["segmentator"], s["train_params"], s["val_params"]])
