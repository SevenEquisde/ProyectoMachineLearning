import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.features import COL_OBJETIVO, COLS_CATEGORICAS, COLS_NUMERICAS, RUTA_TEST, RUTA_TRAIN
    from src.train import RUTA_MEJOR_MODELO, RUTA_TODOS_MODELOS
except ImportError:
    from features import COL_OBJETIVO, COLS_CATEGORICAS, COLS_NUMERICAS, RUTA_TEST, RUTA_TRAIN
    from train import RUTA_MEJOR_MODELO, RUTA_TODOS_MODELOS

RUTA_TABLA_CSV = "docs/resultados_evaluacion.csv"
RUTA_REPORTE_MD = "docs/resultados_evaluacion.md"
RUTA_GRAFICA_PRED = "docs/reales_vs_predichos.png"
RUTA_GRAFICA_RESIDUOS = "docs/distribucion_residuos.png"


def calcular_metricas(y_real, y_pred):
    """Calcula R2, RMSE y MAE (compatible con todas las versiones de sklearn)."""
    r2 = r2_score(y_real, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = mean_absolute_error(y_real, y_pred)
    return r2, rmse, mae


def graficar_reales_vs_predichos(y_test, y_pred, nombre_modelo, ruta_salida):
    """Genera dispersión de precios reales frente a predichos con línea ideal y=x."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5, label="Vehículos (Test)")

    limite_max = max(float(y_test.max()), float(y_pred.max())) * 1.05
    plt.plot([0, limite_max], [0, limite_max], "r--", linewidth=1.5, label="Predicción perfecta (y = x)")

    plt.title(f"Precios Reales vs. Predichos — {nombre_modelo}")
    plt.xlabel("Precio Real (USD)")
    plt.ylabel("Precio Predicho (USD)")
    plt.xlim(0, limite_max)
    plt.ylim(0, limite_max)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def graficar_residuos(y_test, y_pred, nombre_modelo, ruta_salida):
    """Genera histograma de residuos (Error = Real - Predicho)."""
    residuos = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuos, bins=20, edgecolor="black", alpha=0.75)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Error = $0")

    plt.title(f"Distribución de Errores (Residuos) — {nombre_modelo}")
    plt.xlabel("Error: Precio Real - Precio Predicho (USD)")
    plt.ylabel("Cantidad de vehículos")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def evaluar_modelos():
    train_df = pd.read_csv(RUTA_TRAIN)
    test_df = pd.read_csv(RUTA_TEST)

    columnas_x = COLS_NUMERICAS + COLS_CATEGORICAS
    X_train, y_train = train_df[columnas_x], train_df[COL_OBJETIVO]
    X_test, y_test = test_df[columnas_x], test_df[COL_OBJETIVO]

    modelos = joblib.load(RUTA_TODOS_MODELOS)
    modelo_ganador = joblib.load(RUTA_MEJOR_MODELO)

    filas = []
    for nombre, pipeline in modelos.items():
        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)

        tr_r2, tr_rmse, tr_mae = calcular_metricas(y_train, pred_train)
        te_r2, te_rmse, te_mae = calcular_metricas(y_test, pred_test)

        filas.append({
            "Modelo": nombre,
            "Train_R2": round(tr_r2, 4),
            "Train_RMSE": round(tr_rmse, 2),
            "Train_MAE": round(tr_mae, 2),
            "Test_R2": round(te_r2, 4),
            "Test_RMSE": round(te_rmse, 2),
            "Test_MAE": round(te_mae, 2),
        })

    df_resultados = pd.DataFrame(filas).sort_values("Test_RMSE").reset_index(drop=True)

    os.makedirs("docs", exist_ok=True)
    df_resultados.to_csv(RUTA_TABLA_CSV, index=False, encoding="utf-8")

    # Guardar reporte en Markdown para documentación del equipo
    nombre_ganador = type(modelo_ganador.named_steps["modelo"]).__name__
    with open(RUTA_REPORTE_MD, "w", encoding="utf-8") as f:
        f.write("# Resultados de Evaluación en Conjunto de Prueba (Test)\n\n")
        f.write(f"Modelo serializado seleccionado por CV: **{nombre_ganador}**\n\n")
        f.write("```text\n")
        f.write(df_resultados.to_string(index=False))
        f.write("\n```\n")
        f.write("\n")

    # Generar gráficas del modelo serializado (ganador de CV)
    y_pred_ganador = modelo_ganador.predict(X_test)
    graficar_reales_vs_predichos(y_test, y_pred_ganador, nombre_ganador, RUTA_GRAFICA_PRED)
    graficar_residuos(y_test, y_pred_ganador, nombre_ganador, RUTA_GRAFICA_RESIDUOS)

    print("\n=== Evaluación Final (Entrenamiento vs. Prueba) ===")
    print(df_resultados.to_string(index=False))
    print(f"\nReportes y gráficas guardados en docs/:")
    print(f" - {RUTA_TABLA_CSV}")
    print(f" - {RUTA_REPORTE_MD}")
    print(f" - {RUTA_GRAFICA_PRED}")
    print(f" - {RUTA_GRAFICA_RESIDUOS}")

    return df_resultados


if __name__ == "__main__":
    evaluar_modelos()