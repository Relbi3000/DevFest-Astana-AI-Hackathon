## Модельные артефакты и конфиги

Папка: `models/`

### Основные артефакты

- `models/demand_forecast.pkl` — объект модели (boosted/XGBoost/Linear) вместе с метаданными: `feature_cols`, `horizon`, и, при наличии, state_dict Transformer.
- `models/demand_forecast_metrics.json` — метрики MAPE и бэктест-результаты.
- `models/demand_forecast_predictions.csv` — предикты на тестовой выборке.

- `models/demand_ts_transformer.pt` — (опционально) Transformer для sequence-прогноза.

- `models/edge_time_model.pkl`, `models/edge_cost_model.pkl` — модели L2R для предсказания времени и стоимости сегмента.
- `models/edge_time_model_quantiles.pkl`, `models/edge_cost_model_quantiles.pkl` — квантильные модели (если LightGBM доступен).
- `models/learn_to_route_config.json`, `models/learn_to_route_metrics.json` — конфиг и метрики L2R.

- `models/edge_gnn.pt` — GNN state + конфиг (если обучался). `edge_gnn_config.json`, `edge_gnn_metrics.json` — конфиг и метрики.

- `models/transport_classifier.pkl` — классификатор транспортного режима с `feature_cols`.
- `models/transport_classifier_metrics.json` и `transport_confusion_matrix.csv` — метрики и confusion matrix.

- `models/rl_policy.pt` — RL-policy (DQN) с весами и минимальной информацией о размере графа.

### Как используются модели

- `ml/route_optimizer.py` интегрирует L2R/GNN (если доступны) для оценки сегментов в оптимизаторе маршрутов.
- `ml/rl_routing.py` при обучении использует модели L2R как shaping reward (если доступны).
- `tools/report_models.py` собирает и интерпретирует JSON-метрики (health: good/warn/poor).

### Форматы

- Большинство моделей сериализуется через `pickle` (scikit-learn/XGBoost/LightGBM) или `torch.save` для PyTorch-состояний.
- Конфиги хранятся в JSON, включают `feature_cols`, `seed`, `model_type` и служат для воспроизводимости.
