## Архитектура решения

Этот документ описывает высокоуровневую архитектуру проекта, его модули и их взаимодействие.

## Компоненты

- datagenerator2.0b.py

  - Генератор синтетического датасета: узлы (`nodes.csv`), рёбра (`edges.csv`), заказы (`orders.csv`), матрицы расстояний/времени/стоимости, `weather.csv`, `scenarios.csv`.
  - Поддерживает OSM через `osmnx` (опционально), кеширование и настройки семплинга.

- ml/embeddings.py

  - Вычисляет спектральные эмбеддинги (Laplacian) и простые агрегированные признаки центральности.
  - Сохраняет `dataset/node_embeddings.csv` и `dataset/node_features.csv`.

- ml/gnn_models.py

  - Тренировка простого GraphSAGE-подобного GNN для предсказания времени/стоимости ребра.
  - Сохраняет `models/edge_gnn.pt` и `models/edge_gnn_config.json`.

- ml/learn_to_route.py

  - «Learn-to-route» — строит пер-реберные регрессии для предсказания времени и стоимости сегмента, используя order-segmentation.
  - Сохраняет `models/edge_time_model.pkl`, `models/edge_cost_model.pkl` и метрики.

- ml/demand_forecast.py и ml/ts_transformer.py

  - Прогноз спроса: классический boosted/linear подход и опционально sequence Transformer (torch).
  - Артефакты: `models/demand_forecast.pkl`, `models/demand_forecast_metrics.json`, `models/demand_ts_transformer.pt`.

- ml/transport_classifier.py

  - Классификатор выбора транспорта (road/air/combined) на основе заказа, фичей узлов и погоды.
  - Сохраняет `models/transport_classifier.pkl` и метрики/импортанс.

- ml/rl_routing.py

  - RL-модуль (DQN) для выбора следующего хопа на графе; опционально использует L2R/GNN для формирования вознаграждения.
  - Сохраняет `models/rl_policy.pt`.

- ml/route_optimizer.py

  - Базовый оптимизатор маршрутов (k-shortest, multicriteria: время/стоимость/надежность), включает сценарный анализ (стохастика, CVaR), интеграцию GNN/L2R при наличии.

- ml/utils.py

  - Общие утилиты: загрузка CSV (с fallback по путям `datasets`/`dataset`), сборка графа, фичи времени, агрегации погоды.

- tools/report_models.py
  - Сборник простых рутин для вывода текущего качества моделей (читает `models/*.json` и файлы предсказаний).

## Взаимодействия

- Основной сценарий: `run_pipeline.py` вызывает генерацию данных (если нужно), затем последовательно запускает модули: `ml.embeddings` → `ml.gnn_models` → `ml.learn_to_route` → `ml.demand_forecast` → `ml.transport_classifier` → `ml.rl_routing` → `ml.route_optimizer`.
- Все модули используют общие CSV в `dataset/` и сохраняют артефакты в `models/`.
- Модули сделаны устойчивыми к отсутствующим опциям: многие шаги опциональны (torch, LightGBM, osmnx и др.).
