# neuro_trainer

Обучение нейросетей Ultralytics YOLO (классификатор, детектор, сегментатор) с графическим интерфейсом на PySide6.

## Требования

- Python 3.10–3.13 (проверено на 3.12).
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+.
- Для обучения на видеокарте — NVIDIA GPU и драйвер с поддержкой CUDA 12.6 (`nvidia-smi` показывает `CUDA Version: 12.6` или выше).

## Установка

Из корня проекта:

```powershell
poetry install -E gui
```

- `-E gui` — ставит PySide6 и matplotlib для графического интерфейса. Без него доступна только библиотека.
- Виртуальное окружение создаётся в `.venv` внутри проекта (`poetry.toml`).
- torch и torchvision ставятся из репозитория PyTorch со сборкой под CUDA 12.6 (`[[tool.poetry.source]] pytorch-cuda` в `pyproject.toml`). На PyPI для Windows есть только CPU-сборка.

После изменения `pyproject.toml` обновить lock-файл и окружение:

```powershell
poetry lock --no-update
poetry install -E gui
```

В Poetry 2.x вместо `poetry lock --no-update` — просто `poetry lock`.

Проверка, что torch видит видеокарту:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Ожидается `2.7.1+cu126 True`. Если `+cpu` или `False` — обучение пойдёт на процессоре.

## Запуск

```powershell
.venv\Scripts\python.exe run.py
```

или `poetry run python run.py`. В PyCharm — интерпретатор `.venv\Scripts\python.exe`, скрипт `run.py`, рабочий каталог — корень проекта.

## Как пользоваться

1. Выбрать модель (например, `yolo11n.pt` — детектор, `yolo11n-cls.pt` — классификатор, `yolo11n-seg.pt` — сегментатор). Тип задачи определяется по имени модели, можно задать вручную.
2. Указать датасет:
   - детектор / сегментатор — путь к `data.yaml` в формате YOLO;
   - классификатор — каталог с подпапками `train/`, `val/` (внутри — папки по классам).
3. Задать параметры (эпохи, batch, размер изображения и т. д.) и нажать «Старт».
4. Прогресс по эпохам — на вкладке прогресса. «Стоп» завершает обучение после текущей эпохи с сохранением весов.

Параметры, не заданные в окне (lr, momentum и др.), берутся из профиля `fast` / `quality` / `low_vram` для соответствующей задачи (`src/neuro_trainer/presets.py`).

## Где что лежит

| Путь | Что |
|---|---|
| `runs/<classify\|detect\|segment>/<run>/` | Результаты обучения: `weights/best.pt`, `weights/last.pt`, `results.csv`, графики Ultralytics |
| `runs/.../<run>/train_metrics.jsonl` | Метрики по эпохам для вкладки прогресса |
| `runs/.../<run>/final_metrics.json` | Итоговые метрики (последняя строка `results.csv`) |
| `configs/weights/` | Кэш скачанных и своих весов `.pt` |
| `configs/presets/` | Стандартные YAML-пресеты |
| `configs/presets_user/` | Пользовательские YAML-пресеты (кнопки «Загрузить» / «Сохранить») |
| `configs/last_session.yaml` | Настройки последнего запуска |

## Частые проблемы

- **`[ERROR] GUI-зависимости не установлены`** — проект установлен без extra: выполнить `poetry install -E gui`.
- **Обучение идёт на CPU** — в окружении CPU-сборка torch. Выполнить `poetry lock --no-update` и `poetry install -E gui`, затем проверить командой из раздела «Установка».
