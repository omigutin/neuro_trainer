from __future__ import annotations
"""
Каталог параметров обучения — единый источник описания для веб-формы, проверки значений,
YAML-пресетов и сборки аргументов model.train().

Чтобы добавить параметр Ultralytics в интерфейс, достаточно добавить ParamSpec в PARAMS:
форма, подсказка, проверка и передача в train() подхватят его автоматически.
Ключи совпадают с аргументами Ultralytics 8.4 (ultralytics/cfg/default.yaml).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from .tasks import TaskKind


class ParamKind(str, Enum):
    """ Тип значения параметра — определяет виджет формы и приведение типа. """
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    # Размер изображения: число (квадрат) или «HxW»
    IMGSZ = "imgsz"


class ParamGroup(str, Enum):
    """ Группа параметров в форме. Значение — заголовок группы. """
    DEVICE = "Устройство и результаты"
    BASIC = "Параметры обучения"
    OPTIM = "Оптимизация"
    AUGMENT = "Аугментации"
    LOSS = "Веса функции потерь"
    TASK = "Особенности задачи"


# Группы, которые форма показывает в блоке «Доп. настройки» (сворачиваемом)
ADVANCED_GROUPS: tuple[ParamGroup, ...] = (ParamGroup.OPTIM, ParamGroup.AUGMENT, ParamGroup.LOSS, ParamGroup.TASK)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """ Описание одного параметра обучения. """
    # Ключ — совпадает с аргументом model.train() Ultralytics
    key: str
    # Группа в форме
    group: ParamGroup
    # Подпись поля
    label: str
    # Тип значения
    kind: ParamKind
    # Значение по умолчанию (как у Ultralytics, если не указано иное)
    default: Any
    # Всплывающая подсказка: что делает параметр и как его выбирать
    help: str
    # Минимум / максимум для чисел (None — без ограничения)
    min: Optional[float] = None
    max: Optional[float] = None
    # Шаг изменения в поле ввода
    step: Optional[float] = None
    # Допустимые значения для CHOICE
    choices: tuple[str, ...] = ()
    # Для каких задач параметр имеет смысл (пусто — для всех)
    tasks: frozenset[TaskKind] = frozenset()

    def applies_to(self, task: TaskKind) -> bool:
        """ Применим ли параметр к задаче. """
        return not self.tasks or task in self.tasks

    def coerce(self, raw: Any) -> Any:
        """ Привести значение из формы/YAML к типу параметра с проверкой диапазона. """
        try:
            if self.kind is ParamKind.BOOL:
                value: Any = _to_bool(raw)
            elif self.kind is ParamKind.INT:
                if isinstance(raw, bool):
                    raise ValueError("expected integer")
                number = float(raw)
                if number != int(number):
                    raise ValueError("expected integer")
                value = int(number)
            elif self.kind is ParamKind.FLOAT:
                value = float(raw)
            elif self.kind is ParamKind.CHOICE:
                value = str(raw).strip()
                if value not in self.choices:
                    raise ValueError(f"expected one of {', '.join(self.choices)}")
            elif self.kind is ParamKind.IMGSZ:
                value = parse_imgsz(raw)
            else:
                raise ValueError(f"unsupported kind {self.kind}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"{self.key}: invalid value {raw!r} ({e})") from None

        if self.kind in (ParamKind.INT, ParamKind.FLOAT):
            if self.min is not None and value < self.min:
                raise ValueError(f"{self.key}: {value} is less than {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"{self.key}: {value} is greater than {self.max}")
        if self.kind is ParamKind.IMGSZ:
            sides = value if isinstance(value, tuple) else (value,)
            if any(s < 32 or s > 4096 for s in sides):
                raise ValueError(f"{self.key}: each side must be in [32..4096]")
        return value


def _to_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("expected boolean")


def parse_imgsz(raw: Any) -> int | tuple[int, int]:
    """ Число → int, «HxW» / [H, W] → (H, W). """
    if isinstance(raw, bool):
        raise ValueError("expected int or HxW")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise ValueError("expected [H, W]")
        return int(raw[0]), int(raw[1])
    s = str(raw).strip().lower()
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    return int(float(s))


def format_imgsz(value: int | tuple[int, int]) -> str:
    """ (H, W) → «HxW», число → строка. """
    return f"{value[0]}x{value[1]}" if isinstance(value, tuple) else str(value)


_CLS = frozenset({TaskKind.CLASSIFIER})
_SEG = frozenset({TaskKind.SEGMENTATION})
_POSE = frozenset({TaskKind.POSE})
_OBB = frozenset({TaskKind.OBB})
# Задачи с рамками/масками на изображении (не классификация)
_SPATIAL = frozenset({TaskKind.DETECTOR, TaskKind.SEGMENTATION, TaskKind.POSE, TaskKind.OBB})

_P = ParamSpec
_G = ParamGroup
_K = ParamKind

PARAMS: tuple[ParamSpec, ...] = (
    # ---------- Устройство и результаты ----------
    _P("device", _G.DEVICE, "Устройство", _K.CHOICE, "auto",
       "На чём обучать. auto — видеокарта, если доступна, иначе процессор. "
       "cuda — видеокарта NVIDIA, mps — видеокарта Apple, cpu — процессор (очень медленно).",
       choices=("auto", "cuda", "cpu", "mps")),
    _P("workers", _G.DEVICE, "Потоки загрузки", _K.INT, 8,
       "Сколько процессов готовят изображения для обучения. Больше — быстрее подача данных, но больше "
       "оперативной памяти. Обычно 4–8; если падает с нехваткой памяти или на Windows зависает — уменьшить.",
       min=0, max=64, step=1),
    _P("seed", _G.DEVICE, "Seed", _K.INT, 0,
       "Начальное значение генератора случайных чисел. Одинаковый seed + deterministic дают повторяемые результаты.",
       min=0, max=2**31 - 1, step=1),
    _P("deterministic", _G.DEVICE, "Детерминированность", _K.BOOL, True,
       "Использовать детерминированные алгоритмы: результат повторяем при том же seed, но обучение может быть медленнее."),
    _P("amp", _G.DEVICE, "Смешанная точность (AMP)", _K.BOOL, True,
       "Обучение в половинной точности там, где это безопасно: быстрее и меньше видеопамяти. "
       "Отключать только при ошибках NaN в потерях."),
    _P("cache", _G.DEVICE, "Кэш изображений", _K.CHOICE, "false",
       "Держать изображения в оперативной памяти (ram) или на диске в подготовленном виде (disk), "
       "чтобы ускорить эпохи. ram требует памяти примерно с размер датасета.",
       choices=("false", "ram", "disk")),

    # ---------- Параметры обучения ----------
    _P("epochs", _G.BASIC, "Эпохи", _K.INT, 100,
       "Сколько раз модель пройдёт весь датасет. Больше — обычно точнее, но дольше; "
       "лишние эпохи отсекает patience.", min=1, max=10000, step=1),
    _P("patience", _G.BASIC, "Терпение (patience)", _K.INT, 100,
       "Ранняя остановка: прекратить обучение, если метрика на проверочной выборке не улучшалась столько эпох. "
       "0 — не останавливать досрочно.", min=0, max=10000, step=1),
    _P("batch", _G.BASIC, "Размер пакета (batch)", _K.INT, 16,
       "Сколько изображений обрабатывается за один шаг. Ограничен видеопамятью: при ошибке нехватки памяти "
       "уменьшить. -1 — подобрать автоматически под ~60% видеопамяти.", min=-1, max=4096, step=1),
    _P("imgsz", _G.BASIC, "Размер изображения", _K.IMGSZ, 640,
       "Размер стороны входного изображения в пикселях (округляется до кратного 32). "
       "Больше — лучше видны мелкие объекты, но медленнее и больше памяти. "
       "Классификация: обычно 224–384; детекция: 640. Можно HxW, но Ultralytics при обучении берёт большую сторону."),
    _P("rect", _G.BASIC, "Прямоугольные пакеты (rect)", _K.BOOL, False,
       "Собирать пакеты из изображений с похожими пропорциями без дополнения до квадрата. "
       "Экономит память и время, но несовместимо с mosaic (он отключается).", tasks=_SPATIAL),
    _P("multi_scale", _G.BASIC, "Мульти-масштаб", _K.FLOAT, 0.0,
       "Случайно менять размер изображения в пределах ±доли от imgsz на каждом пакете (0.5 = ±50%). "
       "Повышает устойчивость к размеру объектов, но требует больше памяти. 0 — выключено.",
       min=0.0, max=1.0, step=0.05, tasks=_SPATIAL),
    _P("time", _G.BASIC, "Лимит времени, ч", _K.FLOAT, 0.0,
       "Максимальная длительность обучения в часах; если задано (> 0), заменяет число эпох. 0 — без лимита.",
       min=0.0, max=1000.0, step=0.5),
    _P("fraction", _G.BASIC, "Доля датасета", _K.FLOAT, 1.0,
       "Какую часть обучающей выборки использовать (1.0 — всю). Удобно для быстрой пробы на части данных.",
       min=0.01, max=1.0, step=0.05),
    _P("single_cls", _G.BASIC, "Один класс", _K.BOOL, False,
       "Считать все классы одним классом «объект». Для задач, где важно только найти объект, а не его тип.",
       tasks=_SPATIAL),
    _P("freeze", _G.BASIC, "Заморозить слоёв", _K.INT, 0,
       "Заморозить первые N слоёв модели (их веса не обучаются). Ускоряет дообучение на маленьком датасете "
       "и снижает переобучение. 0 — обучать все слои; для YOLO 10 ≈ весь backbone.", min=0, max=100, step=1),
    _P("save_period", _G.BASIC, "Сохранять каждые N эпох", _K.INT, -1,
       "Дополнительно сохранять веса каждые N эпох (epochN.pt). -1 — только best.pt и last.pt.",
       min=-1, max=10000, step=1),

    # ---------- Оптимизация ----------
    _P("optimizer", _G.OPTIM, "Оптимизатор", _K.CHOICE, "auto",
       "Алгоритм обновления весов. auto — Ultralytics выберет сам по длине обучения "
       "(для YOLO26 — MuSGD). SGD — классика для долгого обучения, AdamW — для коротких дообучений.",
       choices=("auto", "SGD", "MuSGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp")),
    _P("lr0", _G.OPTIM, "Начальная скорость (lr0)", _K.FLOAT, 0.01,
       "Начальная скорость обучения. Слишком большая — потери скачут или NaN, слишком маленькая — учится медленно. "
       "SGD ≈ 0.01, Adam/AdamW ≈ 0.001. При optimizer=auto игнорируется.", min=0.0, max=1.0, step=0.001),
    _P("lrf", _G.OPTIM, "Финальный множитель (lrf)", _K.FLOAT, 0.01,
       "Скорость обучения в конце = lr0 × lrf. Меньше — плавнее дообучение на последних эпохах.",
       min=0.0, max=1.0, step=0.001),
    _P("cos_lr", _G.OPTIM, "Косинусное расписание", _K.BOOL, False,
       "Снижать скорость обучения по косинусу вместо линейного снижения. Часто чуть лучше на длинном обучении."),
    _P("momentum", _G.OPTIM, "Momentum", _K.FLOAT, 0.937,
       "Инерция SGD (или beta1 для Adam): насколько учитываются прошлые шаги. Обычно не трогать.",
       min=0.0, max=1.0, step=0.001),
    _P("weight_decay", _G.OPTIM, "Weight decay", _K.FLOAT, 0.0005,
       "L2-регуляризация: штраф за большие веса, снижает переобучение. Обычно 0.0005.",
       min=0.0, max=0.1, step=0.0001),
    _P("warmup_epochs", _G.OPTIM, "Эпох разогрева", _K.FLOAT, 3.0,
       "Сколько первых эпох скорость обучения плавно растёт от малой до lr0. Стабилизирует начало обучения.",
       min=0.0, max=100.0, step=0.5),
    _P("warmup_momentum", _G.OPTIM, "Momentum разогрева", _K.FLOAT, 0.8,
       "Начальный momentum в период разогрева.", min=0.0, max=1.0, step=0.01),
    _P("warmup_bias_lr", _G.OPTIM, "lr смещений при разогреве", _K.FLOAT, 0.1,
       "Скорость обучения для смещений (bias) в период разогрева.", min=0.0, max=1.0, step=0.01),
    _P("nbs", _G.OPTIM, "Номинальный пакет (nbs)", _K.INT, 64,
       "Номинальный размер пакета для нормировки потерь и weight decay. Обычно не трогать.",
       min=1, max=1024, step=1),

    # ---------- Аугментации ----------
    _P("hsv_h", _G.AUGMENT, "Оттенок (hsv_h)", _K.FLOAT, 0.015,
       "Случайный сдвиг оттенка цвета (доля). Помогает при разном освещении и камерах.",
       min=0.0, max=1.0, step=0.005),
    _P("hsv_s", _G.AUGMENT, "Насыщенность (hsv_s)", _K.FLOAT, 0.7,
       "Случайное изменение насыщенности цвета (доля).", min=0.0, max=1.0, step=0.05),
    _P("hsv_v", _G.AUGMENT, "Яркость (hsv_v)", _K.FLOAT, 0.4,
       "Случайное изменение яркости (доля). Полезно, если освещение на объекте меняется.",
       min=0.0, max=1.0, step=0.05),
    _P("degrees", _G.AUGMENT, "Поворот, °", _K.FLOAT, 0.0,
       "Случайный поворот изображения в пределах ±градусов. Для камер с фиксированным ракурсом обычно 0–10.",
       min=0.0, max=180.0, step=1.0),
    _P("translate", _G.AUGMENT, "Сдвиг", _K.FLOAT, 0.1,
       "Случайный сдвиг изображения на ±долю ширины/высоты.", min=0.0, max=0.9, step=0.05),
    _P("scale", _G.AUGMENT, "Масштаб", _K.FLOAT, 0.5,
       "Случайное масштабирование на ±долю. Учит находить объекты разного размера.",
       min=0.0, max=0.9, step=0.05),
    _P("shear", _G.AUGMENT, "Скос, °", _K.FLOAT, 0.0,
       "Случайный скос изображения в пределах ±градусов.", min=0.0, max=45.0, step=0.5),
    _P("perspective", _G.AUGMENT, "Перспектива", _K.FLOAT, 0.0,
       "Случайное перспективное искажение (обычно 0–0.001).", min=0.0, max=0.001, step=0.0001),
    _P("flipud", _G.AUGMENT, "Отражение по вертикали", _K.FLOAT, 0.0,
       "Вероятность отразить изображение сверху вниз. Только если объект может быть «вверх ногами».",
       min=0.0, max=1.0, step=0.05),
    _P("fliplr", _G.AUGMENT, "Отражение по горизонтали", _K.FLOAT, 0.5,
       "Вероятность отразить изображение слева направо. Выключить (0), если важна сторона (лево/право).",
       min=0.0, max=1.0, step=0.05),
    _P("bgr", _G.AUGMENT, "Перестановка RGB↔BGR", _K.FLOAT, 0.0,
       "Вероятность поменять местами каналы RGB и BGR.", min=0.0, max=1.0, step=0.05),
    _P("mosaic", _G.AUGMENT, "Мозаика", _K.FLOAT, 1.0,
       "Вероятность склеить 4 изображения в одно. Сильно помогает детекции мелких объектов; "
       "отключается при rect.", min=0.0, max=1.0, step=0.05, tasks=_SPATIAL),
    _P("close_mosaic", _G.AUGMENT, "Без мозаики последние N эпох", _K.INT, 10,
       "Отключить мозаику на последних N эпохах, чтобы модель доучилась на «естественных» изображениях. "
       "0 — не отключать.", min=0, max=10000, step=1, tasks=_SPATIAL),
    _P("mixup", _G.AUGMENT, "MixUp", _K.FLOAT, 0.0,
       "Вероятность наложить два изображения полупрозрачно. Регуляризация для больших моделей.",
       min=0.0, max=1.0, step=0.05, tasks=_SPATIAL),
    _P("cutmix", _G.AUGMENT, "CutMix", _K.FLOAT, 0.0,
       "Вероятность вставить прямоугольный фрагмент другого изображения.",
       min=0.0, max=1.0, step=0.05, tasks=_SPATIAL),
    _P("copy_paste", _G.AUGMENT, "Copy-paste", _K.FLOAT, 0.0,
       "Вероятность копировать объекты по маскам между изображениями (только сегментация).",
       min=0.0, max=1.0, step=0.05, tasks=_SEG),
    _P("copy_paste_mode", _G.AUGMENT, "Режим copy-paste", _K.CHOICE, "flip",
       "Как вставлять объекты при copy-paste: flip — отражённая копия, mixup — из другого изображения.",
       choices=("flip", "mixup"), tasks=_SEG),
    _P("auto_augment", _G.AUGMENT, "Автоаугментация", _K.CHOICE, "randaugment",
       "Политика автоматических аугментаций для классификации: randaugment (по умолчанию), "
       "autoaugment, augmix.", choices=("randaugment", "autoaugment", "augmix"), tasks=_CLS),
    _P("erasing", _G.AUGMENT, "Случайное стирание", _K.FLOAT, 0.4,
       "Вероятность закрасить случайный участок изображения (классификация). "
       "Учит не опираться на одну деталь.", min=0.0, max=0.9, step=0.05, tasks=_CLS),

    # ---------- Веса функции потерь ----------
    _P("box", _G.LOSS, "Рамки (box)", _K.FLOAT, 7.5,
       "Вес ошибки положения рамки. Больше — точнее рамки, но может ухудшить классификацию.",
       min=0.0, max=100.0, step=0.5, tasks=_SPATIAL),
    _P("cls", _G.LOSS, "Классы (cls)", _K.FLOAT, 0.5,
       "Вес ошибки класса объекта.", min=0.0, max=100.0, step=0.1, tasks=_SPATIAL),
    _P("dfl", _G.LOSS, "DFL", _K.FLOAT, 1.5,
       "Вес distribution focal loss для уточнения границ рамки (у YOLO26 не используется).",
       min=0.0, max=100.0, step=0.1, tasks=_SPATIAL),
    _P("cls_pw", _G.LOSS, "Баланс классов", _K.FLOAT, 0.0,
       "Степень взвешивания редких классов: 0 — выключено, 1 — полностью обратно пропорционально частоте. "
       "Помогает при сильном дисбалансе классов.", min=0.0, max=1.0, step=0.1),
    _P("pose", _G.LOSS, "Ключевые точки (pose)", _K.FLOAT, 12.0,
       "Вес ошибки положения ключевых точек.", min=0.0, max=100.0, step=0.5, tasks=_POSE),
    _P("kobj", _G.LOSS, "Видимость точек (kobj)", _K.FLOAT, 1.0,
       "Вес ошибки видимости ключевых точек.", min=0.0, max=100.0, step=0.1, tasks=_POSE),
    _P("angle", _G.LOSS, "Угол (angle)", _K.FLOAT, 1.0,
       "Вес ошибки угла поворота рамки.", min=0.0, max=100.0, step=0.1, tasks=_OBB),

    # ---------- Особенности задачи ----------
    _P("dropout", _G.TASK, "Dropout", _K.FLOAT, 0.0,
       "Доля случайно отключаемых нейронов в голове классификатора. Снижает переобучение на малых датасетах.",
       min=0.0, max=0.9, step=0.05, tasks=_CLS),
    _P("overlap_mask", _G.TASK, "Слияние масок", _K.BOOL, True,
       "Объединять маски объектов в одну при обучении (быстрее). Выключить, если объекты сильно перекрываются.",
       tasks=_SEG),
    _P("mask_ratio", _G.TASK, "Сжатие масок", _K.INT, 4,
       "Во сколько раз уменьшать маски при обучении. Меньше — точнее контуры, но больше памяти.",
       min=1, max=16, step=1, tasks=_SEG),
)

_BY_KEY: dict[str, ParamSpec] = {p.key: p for p in PARAMS}

if len(_BY_KEY) != len(PARAMS):
    raise RuntimeError("Duplicate keys in PARAMS")


def get_spec(key: str) -> ParamSpec:
    """ Описание параметра по ключу. """
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown parameter '{key}'") from None


def has_param(key: str) -> bool:
    """ Есть ли параметр в каталоге. """
    return key in _BY_KEY


def default_values() -> dict[str, Any]:
    """ Значения по умолчанию для всех параметров каталога. """
    return {p.key: p.default for p in PARAMS}


def coerce_values(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Привести значения к типам каталога.
    Возвращает (значения, ошибки по ключам); неизвестные ключи пропускаются.
    """
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, value in raw.items():
        spec = _BY_KEY.get(key)
        if spec is None:
            continue
        try:
            values[key] = spec.coerce(value)
        except ValueError as e:
            errors[key] = str(e)
    return values, errors


def train_kwargs(values: dict[str, Any], task: TaskKind, keys: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """
    Аргументы model.train() из значений каталога для задачи:
    только применимые к задаче параметры, в формате Ultralytics.
    """
    out: dict[str, Any] = {}
    for spec in PARAMS:
        if keys is not None and spec.key not in keys:
            continue
        if spec.key not in values or not spec.applies_to(task):
            continue
        value = values[spec.key]
        if spec.key == "device":
            if value == "auto":
                continue
        elif spec.key == "cache":
            value = False if value == "false" else value
        elif spec.key == "time":
            if not value:
                continue
        elif spec.key == "freeze":
            if not value:
                continue
        elif spec.key == "imgsz" and isinstance(value, tuple):
            # Ultralytics при обучении принимает только квадрат
            value = max(value)
        out[spec.key] = value
    return out
