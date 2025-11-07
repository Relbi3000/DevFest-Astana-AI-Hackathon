## Подробные инструкции по запуску (Windows PowerShell)

### 1) Подготовка окружения

Рекомендуется виртуальное окружение Python 3.8+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Если у вас GPU и вы хотите использовать PyTorch с CUDA — установите подходящий `torch` вручную по инструкции с https://pytorch.org

### 2) Генерация данных и запуск полного pipeline

`run_pipeline.py` ориентирован на удобный последовательный запуск. Он:

- Генерирует данные (если найден генератор и отсутствуют нужные CSV)
- Вызывает `ml.embeddings`, `ml.gnn_models`, `ml.learn_to_route`, `ml.demand_forecast`, `ml.transport_classifier`, `ml.rl_routing`, `ml.route_optimizer` по очереди

Запуск:

```powershell
python run_pipeline.py
```

Примечание: некоторые шаги (GNN, RL, Transformer) потребуют наличия PyTorch и могут быть пропущены автоматически, если зависимости отсутствуют.

### 3) Запуск отдельных модулей

- Вычисление эмбеддингов и node_features:

```powershell
python -m ml.embeddings
```

- Обучение GNN (если есть torch):

```powershell
python -m ml.gnn_models
```

- Обучение L2R (learn-to-route):

```powershell
python -m ml.learn_to_route
```

- Прогноз спроса (baseline + transformer если есть):

```powershell
python -m ml.demand_forecast
```

- Тренировка классификатора транспорта:

```powershell
python -m ml.transport_classifier
```

- Обучение/инференс RL-политики (требует torch):

```powershell
python -m ml.rl_routing
```

- Демонстрация оптимизатора маршрутов (принтит топ-результат):

```powershell
python -m ml.route_optimizer
```

### 4) Отчёты и качество

Сводный отчёт по моделям:

```powershell
python tools\report_models.py
```

Отдельные JSON-файлы метрик находятся в `models/`:

- `demand_forecast_metrics.json`, `edge_gnn_metrics.json`, `learn_to_route_metrics.json`, `transport_classifier_metrics.json`.

### 5) Советы при проблемах

- Если шаг пропускается, смотрите логи — модули обычно перехватывают ошибки импорта и печатают причину (например, отсутствует `torch` или `lightgbm`).
- Для ускорения обучения используйте GPU для PyTorch и LightGBM (при сборке с поддержкой GPU).
