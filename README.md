# Estimación de Precios de Automóviles Usados — El Salvador (2026)

Modelo supervisado de regresión desarrollado en Python y `scikit-learn` para estimar el precio de venta (USD) de vehículos usados en el mercado salvadoreño.

## Estructura del repositorio
```text
├── data/
│   ├── raw/          # Dataset original (no versionado en Git)
│   └── processed/    # dataset_limpio.csv, train.csv, test.csv
├── docs/             # Diccionario de datos, métricas y gráficas exportadas
├── models/           # Pipeline serializado (pipeline_precios.pkl) y tabla CV
├── notebooks/        # 01_eda_marketplace.ipynb (Análisis Exploratorio)
├── src/              # clean_data.py, features.py, train.py, evaluate.py, predict.py, main.py
└── tests/            # Pruebas unitarias (test_features.py)
