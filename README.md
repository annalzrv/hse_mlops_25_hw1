# ML Fraud Detection Service

Сервис для автоматического обнаружения мошеннических транзакций в режиме батчевого скоринга. Обрабатывает CSV-файлы из указанной директории с использованием обученной CatBoost модели.

## Архитектура решения

```
├── .gitignore
├── Dockerfile
├── README.md
├── app/
│   └── app.py              # Ядро сервиса с обработчиком файлов
├── models/
│   └── my_catboost.cbm     # Обученная модель CatBoost (включена в репо)
├── src/
│   ├── preprocessing.py    # Пайплайн обработки данных
│   └── scorer.py           # Модуль прогнозирования + bonus outputs
├── train_model.py          # Скрипт обучения модели
├── train_data/
│   └── train.csv           # Данные для обучения (скачать из соревнования)
├── input/                  # Директория для загрузки файлов на скоринг
└── output/                 # Директория с результатами скоринга
```

## Выходные файлы

Сервис генерирует следующие файлы в директории `./output`:

| Файл | Описание |
|------|----------|
| `predictions_*.csv` | Предсказания в формате sample_submission (index, prediction) |
| `feature_importances.json` | **[BONUS]** Топ-5 важнейших признаков модели |
| `score_distribution_*.png` | **[BONUS]** График плотности распределения скоров |

### Пример feature_importances.json

```json
{
  "amount_log": 39.65,
  "cat_id_cat": 36.25,
  "hour": 10.82,
  "population_city_log": 3.08,
  "gender_cat": 2.14
}
```

## Пайплайн обработки данных

1. **Загрузка входного файла** — `app/app.py` мониторит директорию `./input`
2. **Препроцессинг** — `src/preprocessing.py`:
   - Временные признаки: час, день недели, месяц, день месяца, год
   - Геопространственные расчеты: расстояние между клиентом и мерчантом
   - Категориальные переменные: группировка редких категорий + mean-encoding
   - Числовые признаки: логарифмирование
3. **Скоринг** — `src/scorer.py`: применение модели CatBoost
4. **Выгрузка** — сохранение результатов в `./output`

## Быстрый старт

### Требования

- Docker 20.10+
- ~2 ГБ свободного места

### 1. Подготовка данных

```bash
# Клонировать репозиторий
git clone https://github.com/annalzrv/hse_mlops_25_hw1.git
cd hse_mlops_25_hw1

# Скачать train.csv из соревнования и поместить в train_data/
# https://www.kaggle.com/competitions/teta-ml-1-2025
cp /path/to/train.csv ./train_data/
```

> **Примечание:** Модель `models/my_catboost.cbm` уже обучена и включена в репозиторий.

### 2. Сборка образа

```bash
docker build -t fraud-detector .
```

### 3. Запуск контейнера

```bash
docker run -it --rm \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    fraud-detector
```

### 4. Скоринг данных

1. Дождитесь сообщения в логах: `File observer started`
2. В **другом терминале** скопируйте `test.csv` в директорию `./input/`:
   ```bash
   cp /path/to/test.csv ./input/
   ```
3. Дождитесь завершения обработки (в логах появится имя выходного файла)
4. Результаты будут в `./output/`:
   - `predictions_*.csv` — предсказания
   - `feature_importances.json` — важность признаков
   - `score_distribution_*.png` — график распределения скоров

## Обучение модели (опционально)

Модель уже обучена и включена в репозиторий. При необходимости можно переобучить:

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить обучение
python train_model.py
```

Результат: `models/my_catboost.cbm` (AUC ~0.997)

## Модельный слой

- **Алгоритм**: CatBoost Classifier
- **Метрика качества**: AUC = 0.997
- **Порог классификации**: 0.98
- **Inference**: CPU-only

## Troubleshooting

**"FileNotFoundError: train.csv"**
- Скачайте `train.csv` с Kaggle и поместите в `./train_data/`

**Сервис не видит новые файлы**
- Убедитесь, что файл имеет расширение `.csv`
- Проверьте, что директория `./input/` правильно примонтирована

**Контейнер падает сразу после запуска**
- Убедитесь, что `train.csv` находится в `./train_data/`

---

*Создано для курса MLOps HSE 2025*
