## Описание датасета

Директория: `dataset/` (или `datasets/` / корень — util `resolve_path` ищет в нескольких местах).

Ключевые файлы (коротко):

- `nodes.csv` — список узлов (склады, хабы, аэропорты).

  - Основные столбцы: `node_id`, `name`, `type` (airport/hub/warehouse), `lat`, `lon`, `operating_start_hour`, `operating_end_hour`, `cutoff_hours`, `service_time_min`, `is_active`.

- `edges.csv` — список рёбер (пары узлов) и мульти-модальных сегментов.

  - Типичные столбцы: `edge_id`, `from_node`, `to_node`, `mode` (`road`/`air`/`transfer`), `distance_km`, `base_time_h`, `base_cost`, `max_weight_kg`, `max_volume_m3`, `reliability`, `co2_kg_per_km`, `schedule_frequency_h`, `cutoff_hours`, `is_active`.

- `orders.csv` — список заказов (реальные или синтетические), основной столбец `order_id`.

  - Включает `origin_id`, `destination_id`, `created_at`, `required_delivery`, `status`, `preferred_mode`, `actual_time_h`, `actual_cost`, `weight_kg`, `volume_m3`, `cargo_class`, `route_nodes` (последовательность `;`), `route_modes` и дополнительные поля (оценки времени/стоимости для разных стратегий).

- `node_features.csv` — агрегированные центральности и эмбеддинги узлов (используется transport classifier и др.).

- `node_embeddings.csv` — спектральные эмбеддинги для узлов (по `ml/embeddings.py`).

- `weather.csv` — табличка по датам/узлам со столбцами: `date`, `node_id`, `weather`/`event`, `time_factor_*`, `cost_factor_*`, `visibility`, `wind` и т.д. Используется для модификации времени/стоимости/надежности.

- `scenarios.csv` — события и сценарии (maintenance, weather_shutdown и пр.) для имитации инцидентов; используется в оптимизаторе маршрутов.

Формат и замечания

- Даты: `created_at` и другие временные поля — обычно ISO-строки; утилиты пытаются парсить `pd.to_datetime`.
- Файлы имеют дублирование имён (например `nodes_test.csv` / `nodes.csv`), `ml/utils.resolve_path` ищет в `datasets`, `dataset`, текущей папке.
- `orders.csv` содержит расширенные метрики и альтернативные стоимости/времена (полезно для оценки базовых стратегий).

Рекомендации по расширению данных

- Для реальной интеграции: заменить синтетический генератор на ETL из реального TMS/WMS/ERP.
- Загрузить реальные расписания авиарейсов в OpenSky (опционально) через `OpenSkyConnector` в `datagenerator2.0b.py`.
