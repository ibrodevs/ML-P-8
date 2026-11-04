# ML-P-8 — Random Forest vs XGBoost

Практическая работа (вехи **8.1** и **8.2**) по оригинальному файлу [o_churn_synthetic.csv](o_churn_synthetic.csv).

## Как запустить

\`\`\`bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_experiment.py
python scripts/make_presentation.py
\`\`\`

Также можно открыть [ML_P8.ipynb](ML_P8.ipynb) и выполнить ячейки сверху вниз.

## Что сделано

- 3000 абонентов, 11 входных признаков без \`subscriber_code\`, целевая переменная \`churn\`.
- Разбиение: **train 2400 / test 600** (80/20), \`stratify=y\`, \`random_state=42\`.
- \`OneHotEncoder\` для \`region\` и \`tariff\`; **отдельный** \`make_prep()\` в каждом Pipeline.
- **Random Forest**: 100 деревьев.
- **XGBoost**: 300 деревьев, learning rate 0.05, max depth 4.
- F1 train, F1 test, ROC-AUC test, время обучения, разрыв F1, confusion matrices.
- Генератор **7-слайдовой презентации**: 3 слайда о каждом методе + сравнительный вывод.

## Фактические результаты

После запуска команды \`python scripts/run_experiment.py\` появятся:
- [outputs/metrics.csv](outputs/metrics.csv), [outputs/metrics.json](outputs/metrics.json), [outputs/report.md](outputs/report.md);
- [outputs/comparison.png](outputs/comparison.png) и confusion matrices;
- после генерации — [presentation/ML_P8_Comparison.pptx](presentation/ML_P8_Comparison.pptx).

Если GitHub Actions разрешено записывать в репозиторий, отчет и PPTX сформируются автоматически после push.

## Ожидаемые результаты с фото (ориентиры, не финальный запуск)

| Модель | F1 train | F1 test | ROC-AUC test |
|---|---:|---:|---:|
| Random Forest | 1.000 | 0.556 | 0.752 |
| XGBoost | 0.789 | **0.574** | **0.779** |

По фотографиям **лучше XGBoost**: выше тестовые F1 и ROC-AUC, меньше разрыв train/test. Для реального вывода используйте **рассчитанные** метрики в \`outputs/metrics.csv\`. Время зависит от оборудования. Для внедрения нужны кросс-валидация и подбор порога.
