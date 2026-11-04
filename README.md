# ML-P-8: сравнение четырёх методов для прогноза оттока

Используются Random Forest, XGBoost, LightGBM и CatBoost на исходном o_churn_synthetic.csv. Вехи 8.1, 8.2 и открытое задание уровня C.

## Данные и методика

- CSV: 3000 абонентов. Train: 2400, test: 600. Разделение 80/20 с stratify=y и random_state=42.
- Цель churn (1 — ушёл, 0 — остался), subscriber_code исключён: 11 входных признаков.
- Для Random Forest, XGBoost, LightGBM: независимый Pipeline с OneHotEncoder для region и tariff.
- Для CatBoost: region и tariff передаются без one-hot, как нативные категориальные признаки.
- Все модели обучаются с n_jobs=1 (CatBoost thread_count=1). Время: 5 повторов после первого прогрева, медиана. Инференс: 600 тестовых строк, медиана 5 повторов. Размер: pickle полных моделей/Pipeline.

## Параметры

| Метод | Параметры | Категории |
|---|---|---|
| Random Forest | n_estimators=100, random_state=42 | One-hot |
| XGBoost | n_estimators=300, learning_rate=0.05, max_depth=4 | One-hot |
| LightGBM | n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15 | One-hot |
| CatBoost | iterations=300, learning_rate=0.05, depth=4 | Native |

## Итоговая сравнительная таблица

<!-- METRICS_START -->
Метрики появятся после выполнения скрипта; GitHub Actions вычисляет их автоматически.
<!-- METRICS_END -->

**Рекомендация:** основной показатель для открытого задания — ROC-AUC test. Также смотрим F1 test, gap train/test, обучение, инференс, размер модели. Лучшая модель по одному показателю не обязательно лучшая по всем.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS; Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_experiment.py
python scripts/make_presentation.py
jupyter notebook
```

## Состав проекта

- [ML_P8.ipynb](ML_P8.ipynb) — Jupyter Notebook с версиями библиотек и отдельными расчётами по каждой модели.
- [scripts/run_experiment.py](scripts/run_experiment.py) — обучение, метрики, сравнение, графики.
- [outputs/metrics.csv](outputs/metrics.csv) и [outputs/report.md](outputs/report.md) — готовые результаты.
- [presentation/ML_P8_Comparison.pptx](presentation/ML_P8_Comparison.pptx) — **12 слайдов: по три на каждый из 4 методов**, последний включает общий вывод.
- [presentation/speaker_notes.md](presentation/speaker_notes.md) — текст выступления.

## Перед сдачей

- [x] Все модели обучены на одинаковом разделении c random_state=42.
- [x] Есть таблица train/test метрик, времени обучения, инференса и размера.
- [x] Версии библиотек фиксируются в outputs/metrics.json и выводятся в Notebook.
- [ ] Проверьте в своей системе выполнение Notebook через Restart & Run All.
