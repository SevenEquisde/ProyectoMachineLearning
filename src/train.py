import os
import joblib
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Compatible ejecutando tanto `python src/train.py` como `python -m src.train`
try:
    from src.features import (
        COL_OBJETIVO,
        COLS_CATEGORICAS,
        COLS_NUMERICAS,
        RUTA_TRAIN,
        construir_preprocesador,
    )
except ImportError:
    from features import (
        COL_OBJETIVO,
        COLS_CATEGORICAS,
        COLS_NUMERICAS,
        RUTA_TRAIN,
        construir_preprocesador,
    )

RUTA_MEJOR_MODELO = "models/pipeline_precios.pkl"
RUTA_TODOS_MODELOS = "models/modelos_entrenados.pkl"
RUTA_RESULTADOS_CV = "models/resultados_cv.csv"


def entrenar_modelos(ruta_train: str = RUTA_TRAIN):
    train_df = pd.read_csv(ruta_train)
    X_train = train_df[COLS_NUMERICAS + COLS_CATEGORICAS]
    y_train = train_df[COL_OBJETIVO]

    # Esquema uniforme de validación cruzada para todos los algoritmos
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    metricas_cv = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    # Familias de modelos e hiperparámetros a explorar
    configuracion_modelos = {
        "Regresion_Lineal": (LinearRegression(), {}),
        "Ridge": (
            Ridge(random_state=42),
            {"modelo__alpha": [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
        ),
        "Lasso": (
            Lasso(max_iter=10000, random_state=42),
            {"modelo__alpha": [1.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]},
        ),
        "Arbol_Decision": (
            DecisionTreeRegressor(random_state=42),
            {
                "modelo__max_depth": [3, 4, 5, 6, 8, 10],
                "modelo__min_samples_leaf": [3, 5, 8, 12, 15],
                "modelo__min_samples_split": [2, 5, 10],
            },
        ),
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, (estimador, param_grid) in configuracion_modelos.items():
        pipeline = Pipeline([
            ("preprocesador", construir_preprocesador()),
            ("modelo", estimador),
        ])

        if param_grid:
            busqueda = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                refit=True,
                n_jobs=-1,
            )
            busqueda.fit(X_train, y_train)
            mejor_pipeline = busqueda.best_estimator_
            mejores_params = busqueda.best_params_
        else:
            pipeline.fit(X_train, y_train)
            mejor_pipeline = pipeline
            mejores_params = {}

        # Evaluar las 3 métricas con el mismo KFold sobre entrenamiento
        scores = cross_validate(mejor_pipeline, X_train, y_train, cv=cv, scoring=metricas_cv)
        r2_cv = scores["test_r2"].mean()
        rmse_cv = -scores["test_rmse"].mean()
        mae_cv = -scores["test_mae"].mean()

        modelos_entrenados[nombre] = mejor_pipeline
        resultados.append({
            "Modelo": nombre,
            "CV_R2": round(r2_cv, 4),
            "CV_RMSE": round(rmse_cv, 2),
            "CV_MAE": round(mae_cv, 2),
            "Mejores_Hiperparametros": str(mejores_params),
        })

    # Ordenar por menor RMSE en validación cruzada
    df_resultados = pd.DataFrame(resultados).sort_values("CV_RMSE").reset_index(drop=True)
    nombre_ganador = df_resultados.loc[0, "Modelo"]
    pipeline_ganador = modelos_entrenados[nombre_ganador]

    # Guardar artefactos
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline_ganador, RUTA_MEJOR_MODELO)
    joblib.dump(modelos_entrenados, RUTA_TODOS_MODELOS)
    df_resultados.to_csv(RUTA_RESULTADOS_CV, index=False, encoding="utf-8")

    print("\n=== Resultados de Validación Cruzada (5-Fold CV) ===")
    print(df_resultados.to_string(index=False))
    print(f"\nModelo ganador seleccionado por menor CV_RMSE: {nombre_ganador}")
    print(f"Pipeline guardado en: {RUTA_MEJOR_MODELO}")

    return df_resultados, modelos_entrenados


if __name__ == "__main__":
    entrenar_modelos()